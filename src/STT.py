from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias
from enum import Enum, auto
from copy import deepcopy
from STT_program import Instruction_Name, Instruction, Program, random_program, print_program
import random

# abstract branch predictor
class BrPr:
    def __init__(self, correct_predictions: Dict, program_length: int):
        self.correct_predictions: Dict = correct_predictions
        self.program_length: int = program_length

    def first_update(self, b: bool) -> BrPr:
        # "The history register (global or local) is updated with the predicted direction b."
        # NOTE: currently does nothing but exists to match spec of STT paper
        return self
    
    def second_update(self, pc: int, cond: bool, target: int) -> BrPr:
        # "predictor is updated with both correct direction cond and correct target address target"
        
        new_bp = deepcopy(self)

        new_bp.correct_predictions[pc] = tuple([target, cond])

        return new_bp
    
    def predict(self, pc_br: int) -> Tuple[int, bool]:
        # "bp.predict(pc_br) = (spc, b), for a given program counter pc_br , the function returns spc, which is the predicted target address, and b, which is a boolean value indicating whether the branch is predicted taken or not"

        if pc_br in self.correct_predictions:
            return self.correct_predictions[pc_br]
        else:
            spc: int = random.randint(0, self.program_length - 1)
            b: bool = random.choice([True, False])

            return tuple([spc, b])

@dataclass
class ReorderBuffer:
    seq     : List[Tuple[int, Instruction, bool]]
    head    : int

# TODO: change to dataclass like `ReorderBuffer`?
class State_STT:
    def __init__(self,
                pc: int,
                mem: Dict,
                reg: Dict,
                ready: Dict,
                rt: Dict,
                rob: ReorderBuffer,
                lq: List[Tuple[int, bool, int]],
                sq: List[int],
                bp: BrPr,
                ckpt: List[Tuple[int, int, Dict]],
                C: List[int],
                T: Dict, # TODO: define taint. Seems like a mapping: PRegID -> bool would do
                ### custom fields ###
                next_fresh: int = 0 # the lowest unused PRegID. TODO: remove if necessary
                ):
        self.pc = pc
        self.mem = mem
        self.reg = reg
        self.ready = ready
        self.rt = rt
        self.rob = rob
        self.lq = lq
        self.sq = sq
        self.bp = bp
        self.ckpt = ckpt
        self.C = C
        self.T = T
        # ...
        self.next_fresh = next_fresh
        return

    def __str__(self):
        s  = f'---- State_STT ------------'
        s += f'\n\tprogram counter: {self.pc}'
        s += f'\n\tmemory: {self.mem}'
        s += f'\n\tregisters: {self.reg}'
        s += f'\n\tready: {self.ready}'
        s += f'\n\trenaming table: {self.rt}'

        rob_seq_s = '['
        for pc, instruction, branch_pred in self.rob.seq:
            rob_seq_s += f'\n\t\t({pc},\t{str(instruction)},\t{str(branch_pred)}), '
        rob_seq_s += '\n\t]'

        s += f'\n\treorder buffer (head: {self.rob.head}): {rob_seq_s}'
        s += f'\n\tload queue: {self.lq}'
        s += f'\n\tstore queue: {self.sq}'

        corr_preds_s = ''
        for pc_br, pred in self.bp.correct_predictions.items():
            corr_preds_s += f'\t{pc_br} : {bool(pred[1])} -> {pred[0]}'

        s += f'\n\tbranch predictor:\n{corr_preds_s}'
        s += f'\n\tbranch checkpoints: {self.ckpt}'
        s += f'\n\tcache: {self.C}'
        s += f'\n\ttaint: {self.T}'
        s += f'\n---------------------------'

        return s



# TODO: need to decide whether or not static and dynamic instructions should be different classes
def rename(state: State_STT, static_instruction: Instruction) -> Tuple[Dict, Instruction]:
    match static_instruction.name:
        case Instruction_Name.IMMED:
            # deepcopy to avoid changing the original renaming table
            new_rt: Dict = deepcopy(state.rt)

            r_d: int = static_instruction.operands[0]
            k: int = static_instruction.operands[1]

            # map logical register to *fresh* physical register
            x_d: int = fresh(state)
            new_rt[r_d] = x_d

            # create a new 'dynamic' instruction with the new physical register as it's first operand
            dynamic_instruction: Instruction = Instruction(Instruction_Name.IMMED, [x_d, k])
            
            return (new_rt, dynamic_instruction)

        case Instruction_Name.OP:
            new_rt: Dict = deepcopy(state.rt)

            r_d: int = static_instruction.operands[0]
            r_a: int = static_instruction.operands[1]
            r_b: int = static_instruction.operands[2]

            x_d: int = fresh(state)
            new_rt[r_d] = x_d

            # map LRegIDs of source registers to their respective PRegIDs from the renaming table
            x_a: int = state.rt[r_a]
            x_b: int = state.rt[r_b]

            dynamic_instruction: Instruction = Instruction(Instruction_Name.OP, [x_d, x_a, x_b])

            return (new_rt, dynamic_instruction)

        case Instruction_Name.BRANCH:
            # rt is unchanged by branch
            rt_copy: Dict = deepcopy(state.rt)

            r_c: int = static_instruction.operands[0]
            r_d: int = static_instruction.operands[1]

            x_c: int = state.rt[r_c]
            x_d: int = state.rt[r_d]

            dynamic_instruction: Instruction = Instruction(Instruction_Name.BRANCH, [x_c, x_d])

            return (rt_copy, dynamic_instruction)

        case Instruction_Name.LOAD:
            new_rt: Dict = deepcopy(state.rt)

            r_d: int = static_instruction.operands[0]
            r_a: int = static_instruction.operands[1]

            x_d: int = fresh(state)
            new_rt[r_d] = x_d

            x_a: int = state.rt[r_a]

            dynamic_instruction: Instruction = Instruction(Instruction_Name.LOAD, [x_d, x_a])

            return (new_rt, dynamic_instruction)

        case Instruction_Name.STORE:
            # rt is unchanged by store
            rt_copy: Dict = deepcopy(state.rt)

            r_a: int = static_instruction.operands[0]
            r_v: int = static_instruction.operands[1]

            x_a: int = state.rt[r_a]
            x_v: int = state.rt[r_v]

            dynamic_instruction: Instruction = Instruction(Instruction_Name.STORE, [x_a, x_v])

            return (rt_copy, dynamic_instruction)

# TODO: implement (STT formal pg. 5)
# TODO: does storing fresh data in state make sense if their are multiple states? are there multiple states??
# "for a given STT state `σ`, returns a physical register `x` that is fresh in `σ`, meaning that it has not been previously mapped to by `rt`"
def fresh(state: State_STT) -> int:
    # keep track of the lowest unused PRegID in state? Then fresh can just return that ID and increment the number
    # make sure that freshest isn't holding a reference to state.next_fresh. deepcopy?
    freshest = state.next_fresh
    state.next_fresh += 1
    return freshest
    
    ### return

# underSpec(ckpt, i) = (∃j. (j,_,_) ∈ ckpt ∧ j < i)
def underSpec(ckpt: List[Tuple[int, int, Dict]], i: int) -> bool:
    for checkpoint in ckpt:
        if checkpoint[0] < i:
            return True
    return False

# TODO: change if taint implementation changes
def taint(state: State_STT, static_instruction: Instruction) -> Dict:
    _, dynamic_instruction = rename(state, static_instruction)
    tainted = state.T

    match dynamic_instruction.name:
        case Instruction_Name.OP:
            new_taint = deepcopy(state.T)
            
            x_d: int = dynamic_instruction.operands[0]
            x_a: int = dynamic_instruction.operands[1]
            x_b: int = dynamic_instruction.operands[2]

            new_taint[x_d] = (x_a in tainted and tainted[x_a]) or (x_b in tainted and tainted[x_b])
            
            return new_taint
        
        case Instruction_Name.LOAD:
            new_taint = deepcopy(state.T)

            x_d: int = dynamic_instruction.operands[0]

            new_taint[x_d] = underSpec(state.ckpt, len(state.rob.seq)) # underSpec(σ.ckpt, σ.rob_tail)

            return new_taint

        case _:
            taint_copy = deepcopy(state.T)
            return taint_copy

# TODO: change if taint implementation changes
def noTaintedInputs(state: State_STT, i: int) -> bool:
    # TODO: bounds check?
    instruction: Instruction = state.rob.seq[i][1]
    tainted: Dict = state.T

    match instruction.name:
        case Instruction_Name.IMMED:
            return True

        case Instruction_Name.OP:
            x_a: int = instruction.operands[1]
            x_b: int = instruction.operands[2]

            return (x_a not in tainted or not tainted[x_a]) and (x_b not in tainted or not tainted[x_b])
        
        case Instruction_Name.LOAD:
            x_a: int = instruction.operands[1]

            return x_a not in tainted or not tainted[x_a]
        
        case Instruction_Name.BRANCH:
            x_c: int = instruction.operands[0]
            x_d: int = instruction.operands[1]

            return (x_c not in tainted or not tainted[x_c]) and (x_d not in tainted or not tainted[x_d])
        
        case Instruction_Name.STORE:
            x_a: int = instruction.operands[0]
            x_v: int = instruction.operands[1]

            return (x_a not in tainted or not tainted[x_a]) and (x_v not in tainted or not tainted[x_v])


# TODO: change if taint implementation changes
def untaintInstr(state: State_STT, i: int) -> Dict:
    instruction: Instruction = state.rob.seq[i][1]

    x_d: int = instruction.operands[0]

    if instruction.name is Instruction_Name.OP and noTaintedInputs(state, i):
        new_taint = deepcopy(state.T)

        new_taint[x_d] = False

        return new_taint
    
    elif instruction.name is Instruction_Name.LOAD and not underSpec(state.ckpt, i):
        new_taint = deepcopy(state.T)

        new_taint[x_d] = False

        return new_taint
    
    else:
        return deepcopy(state.T)

def untaint(state: State_STT) -> State_STT:
    new_state = deepcopy(state)

    for i in range(state.rob.head, len(state.rob.seq)-1): # 'for i from σ.rob_head to σ.rob_tail − 1 do...'
        new_state.T = untaintInstr(new_state, i)
    
    return new_state

# μ-events

### fetch events

def fetch_immediate(state: State_STT, static_instruction: Instruction) -> State_STT:
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_arithmetic(state: State_STT, static_instruction: Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_branch(state: State_STT, static_instruction: Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(state, static_instruction)

    spc, b = state.bp.predict(state.pc)

    new_bp = new_state.bp.first_update(b)

    new_pc = spc if b else state.pc + 1

    new_ckpt = tuple([len(state.rob.seq), state.pc, state.rt]) # "newCkpt = (rob_tail, pc, rt)"

    new_state.pc = new_pc

    new_state.rt = new_rt

    spc_b = tuple([spc, b])
    rob_entry = tuple([state.pc, dynamic_instruction, spc_b])
    new_state.rob.seq.append(rob_entry)

    new_state.bp = new_bp

    new_state.ckpt.append(new_ckpt)

    return new_state

def fetch_load(state: State_STT, static_instruction: Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_store(state: State_STT, static_instruction: Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(state, static_instruction)

    new_state.pc = state.pc + 1

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    new_state.sq.append(len(state.rob.seq)) #  "sq ++ [rob_tail]"

    return new_state

### execute events

#### load helper functions

def StoresAreReady(state: State_STT, rob_i: int) -> bool:
    for rob_j in state.sq:
        if rob_j < rob_i:
            _, older_store, _ = state.rob.seq[rob_j]

            assert(older_store.name == Instruction_Name.STORE)

            x_b: int = older_store.operands[0]

            if not state.ready[x_b]:
                return False

    return True

def CanForward(state: State_STT, rob_i: int, x_a: int, rob_j: int, x_v: int) -> bool:
    in_store_queue: bool = rob_j in state.sq
    is_younger: bool = rob_j < rob_i

    _, dynamic_instruction, _ = state.rob.seq[rob_j]

    assert(dynamic_instruction.name == Instruction_Name.STORE)

    x_b: int = dynamic_instruction.operands[0]

    registers_ready: bool = state.ready[x_b] and state.ready[x_v]

    addresses_match: bool = state.reg[x_a] == state.reg[x_b]

    no_younger_stores: bool = True
    for rob_k in state.sq:
        if rob_j < rob_k < rob_i:
            _, younger_instruction, _ = state.rob.seq[rob_k]

            k_v: int = younger_instruction.operands[1]

            if CanForward(state, rob_i, x_a, rob_k, k_v):
                no_younger_stores = False
                break


    return in_store_queue and is_younger and registers_ready and addresses_match and no_younger_stores

def loadResult(state: State_STT, rob_i: int, x_a: int, oldRes: int) -> int:
    for rob_j in state.sq:
        if rob_j < rob_i:
            _, dynamic_instruction, _ = state.rob.seq[rob_j]

            x_v: int = dynamic_instruction.operands[1]

            if CanForward(state, rob_i, x_a, rob_j, x_v):
                return state.reg[x_v]
    
    return oldRes

# TODO: decide how to implement (STT formal pg. 8)
# "for a given sequence of memory addresses already accessed and the next address to access returns the number of cycles it takes to load the value by that address"
def LoadLat(cache: List[int], target_address: int) -> int:
    return 1


####

def execute_immediate(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_immediate(state, rob_index)) # result is undefined if event isn't enabled

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    k: int = dynamic_instruction.operands[1]

    new_state.reg[x_d] = k
    new_state.ready[x_d] = True

    return new_state

def enabled_execute_immediate(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is True:
        return False
    
    return True

def execute_arithmetic(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_arithmetic(state, rob_index))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]
    x_b: int = dynamic_instruction.operands[2]

    new_state.reg[x_d] = state.reg[x_a] + state.reg[x_b] # Assuming that `op` is always addition

    new_state.ready[x_d] = True

    return new_state

def enabled_execute_arithmetic(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]
    x_b: int = dynamic_instruction.operands[2]

    if x_d not in state.ready or state.ready[x_d] is True:
        return False
    if x_a not in state.ready or state.ready[x_a] is False:
        return False
    if x_b not in state.ready or state.ready[x_b] is False:
        return False

    return True

def execute_branch_success(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_branch_success(state, rob_index))

    new_state = deepcopy(state)

    origin_pc, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    new_bp = new_state.bp.second_update(origin_pc, state.reg[x_c], state.reg[x_d])
    new_state.bp = new_bp

    new_state.ckpt = [ checkpoint for checkpoint in state.ckpt if checkpoint[0] != rob_index ] #  "ckpt | != i"

    return new_state

def enabled_execute_branch_success(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, branch_prediction = state.rob.seq[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    if x_c not in state.ready or state.ready[x_c] is False:
        return False
    if x_d not in state.ready or state.ready[x_d] is False:
        return False
    if b != state.reg[x_c]:
        return False
    if spc != state.reg[x_d]:
        return False

    return True

def execute_branch_fail(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_branch_fail(state, rob_index))

    new_state = deepcopy(state)

    origin_pc, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    new_pc = state.reg[x_d] if state.reg[x_c] else origin_pc + 1
    new_state.pc = new_pc
    
    rollback_ckpt: Tuple[int, int, Dict] = None

    for checkpoint in state.ckpt:
        if checkpoint[0] == rob_index and checkpoint[1] == origin_pc:
            rollback_ckpt = checkpoint
            break
    
    assert(rollback_ckpt is not None)

    new_rt = rollback_ckpt[2]
    new_state.rt = new_rt

    new_rob_seq = state.rob.seq[:rob_index]
    new_state.rob.seq = new_rob_seq

    new_lq = [ queued_load for queued_load in state.lq if queued_load[0] < rob_index ]
    new_state.lq = new_lq

    new_sq = [ queued_store for queued_store in state.sq if queued_store < rob_index ]
    new_state.sq = new_sq

    new_bp = new_state.bp.second_update(origin_pc, state.reg[x_c], state.reg[x_d])
    new_state.bp = new_bp

    new_ckpt = [ checkpoint for checkpoint in state.ckpt if checkpoint[0] < rob_index ]
    new_state.ckpt = new_ckpt

    return new_state

def enabled_execute_branch_fail(state: State_STT, rob_index: int) -> bool:
    origin_pc, dynamic_instruction, branch_prediction = state.rob.seq[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    if x_c not in state.ready or state.ready[x_c] is False:
        return False
    if x_d not in state.ready or state.ready[x_d] is False:
        return False
    if b == state.reg[x_c] and spc == state.reg[x_d]:
        return False

    rollback_ckpt: Tuple[int, int, Dict] = None
    for checkpoint in state.ckpt:
        if checkpoint[0] == rob_index and checkpoint[1] == origin_pc:
            rollback_ckpt = checkpoint
            break
    if rollback_ckpt is None:
        return False
    
    return True

def execute_load_begin_get_s(state: State_STT, rob_index: int, t: int) -> State_STT:
    assert(enabled_execute_load_begin_get_s(state, rob_index))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_a: int = dynamic_instruction.operands[1]

    t_end = t + LoadLat(state.C, state.reg[x_a])

    lq_entry = None
    lq_index = None
    for i, entry in enumerate(state.lq):
        print(i)
        print(entry)
        if entry[0] == rob_index:
            assert(entry[2] is None) # might be unnecessary
            lq_entry = deepcopy(entry)
            lq_index = i
            break

    # TODO: Make sure this is correct (event spec seems to suggest that the entry should *already* exist and is merely modified by the event)
    if lq_entry is None:
        lq_entry = tuple([rob_index, False, None])
    new_state.lq.append(lq_entry)
    lq_index = len(new_state.lq) - 1

    lq_entry = list(lq_entry)
    lq_entry[2] = t_end # (i, False, ⊥) -> (i, False, t_end)
    lq_entry = tuple(lq_entry)
    new_state.lq[lq_index] = lq_entry

    new_state.C.append(state.reg[x_a])

    return new_state

def enabled_execute_load_begin_get_s(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    if x_a not in state.ready or state.ready[x_a] is False:
        return False
    if x_d not in state.ready or state.ready[x_d] is True:
        return False
    if not StoresAreReady(state, rob_index):
        return False

    return True

def execute_load_end_get_s(state: State_STT, rob_index: int, t: int) -> State_STT:
    assert(enabled_execute_load_end_get_s(state, rob_index, t))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    lq_entry = None
    lq_index = None
    for i, entry in enumerate(state.lq):
        if entry[0] == rob_index:
            lq_entry = deepcopy(entry)
            lq_index = i
            break

    assert(lq_entry is not None and lq_index is not None)

    new_state.reg[x_d] = state.mem[state.reg[x_a]]

    lq_entry = list(lq_entry)
    lq_entry[1] = True # (i, False, t_end) -> (i, True, t_end)
    lq_entry = tuple(lq_entry)
    new_state.lq[lq_index] = lq_entry

    return new_state

def enabled_execute_load_end_get_s(state: State_STT, rob_index: int, t: int) -> bool:
    lq_entry = None
    for entry in state.lq:
        if entry[0] == rob_index:
            lq_entry = deepcopy(entry)
            break
    
    assert(lq_entry is not None)

    t_end = lq_entry[2]

    if t_end is None:
        return False
    if t_end < t:
        return False

    return True

def execute_load_complete(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_load_complete(state, rob_index))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    res: int = loadResult(state, rob_index, x_a, state.reg[x_d])

    new_state.reg[x_d] = res

    new_state.ready[x_d] = True

    return new_state

def enabled_execute_load_complete(state: State_STT, rob_index: int) -> bool:
    lq_entry = None
    for entry in state.lq:
        if entry[0] == rob_index and entry[1] is True: # (i, True, _) ∈ lq
            lq_entry = entry
            break


    return lq_entry is not None

### commit events

def commit_immediate(state: State_STT) -> State_STT:
    assert(enabled_commit_immediate(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    return new_state

def enabled_commit_immediate(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        return False

    return True

def commit_arithmetic(state: State_STT) -> State_STT:
    assert(enabled_commit_arithmetic(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    return new_state

def enabled_commit_arithmetic(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        return False

    return True

def commit_branch(state: State_STT) -> State_STT:
    assert(enabled_commit_branch(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    return new_state

def enabled_commit_branch(state: State_STT) -> bool:
    checkpoint_indices = [ checkpoint[0] for checkpoint in state.ckpt ]

    if state.rob.head in checkpoint_indices:
        return False

    return True

def commit_load(state: State_STT) -> State_STT:
    assert(enabled_commit_load(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    new_state.lq = [ entry for entry in state.lq if entry[0] != state.rob.head ]

    return new_state

def enabled_commit_load(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        return False

    return True

def commit_store(state: State_STT) -> State_STT:
    assert(enabled_commit_store(state))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[state.rob.head]

    x_a: int = dynamic_instruction.operands[0]
    x_v: int = dynamic_instruction.operands[1]

    new_state.mem[state.reg[x_a]] = state.reg[x_v]

    new_state.rob.head += 1

    new_state.sq = [ entry for entry in state.sq if entry != state.rob.head ]

    new_state.C.append(state.reg[x_a])

    return new_state

def enabled_commit_store(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob.seq[state.rob.head]

    x_a: int = dynamic_instruction.operands[0]
    x_v: int = dynamic_instruction.operands[1]

    if x_a not in state.ready or state.ready[x_a] is False:
        return False
    if x_v not in state.ready or state.ready[x_v] is False:
        return False

    return True

###

class M_Event_Name(Enum):
    FETCH_IMMEDIATE             = auto()
    FETCH_ARITHMETIC            = auto()
    FETCH_BRANCH                = auto()
    FETCH_LOAD                  = auto()
    FETCH_STORE                 = auto()
    EXECUTE_IMMEDIATE           = auto()
    EXECUTE_ARITHMETIC          = auto()
    EXECUTE_BRANCH_SUCCESS      = auto()
    EXECUTE_BRANCH_FAIL         = auto()
    EXECUTE_LOAD_BEGIN_GET_S    = auto()
    EXECUTE_LOAD_END_GET_S      = auto()
    EXECUTE_LOAD_COMPLETE       = auto()
    COMMIT_IMMEDIATE            = auto()
    COMMIT_ARITHMETIC           = auto()
    COMMIT_BRANCH               = auto()
    COMMIT_LOAD                 = auto()
    COMMIT_STORE                = auto()

@dataclass
class M_Event:
    name        : M_Event_Name
    rob_index   : Optional[int]
    instruction : Optional[Instruction]

    def __init__(self, name: M_Event_Name, rob_index: Optional[int], instruction : Optional[Instruction]):
        self.name = name
        self.rob_index = rob_index
        self.instruction = instruction

def perform(state: State_STT, event: M_Event, t: int) -> State_STT:
    match event.name:
        case M_Event_Name.FETCH_IMMEDIATE:
            return fetch_immediate(state, event.instruction)

        case M_Event_Name.FETCH_ARITHMETIC:
            return fetch_arithmetic(state, event.instruction)

        case M_Event_Name.FETCH_BRANCH:
            return fetch_branch(state, event.instruction)

        case M_Event_Name.FETCH_LOAD:
            return fetch_load(state, event.instruction)

        case M_Event_Name.FETCH_STORE:
            return fetch_store(state, event.instruction)

        case M_Event_Name.EXECUTE_IMMEDIATE:
            return execute_immediate(state, event.rob_index)

        case M_Event_Name.EXECUTE_ARITHMETIC:
            return execute_arithmetic(state, event.rob_index)

        case M_Event_Name.EXECUTE_BRANCH_SUCCESS:
            return execute_branch_success(state, event.rob_index)

        case M_Event_Name.EXECUTE_BRANCH_FAIL:
            return execute_branch_fail(state, event.rob_index)

        case M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S:
            return execute_load_begin_get_s(state, event.rob_index, t)

        case M_Event_Name.EXECUTE_LOAD_END_GET_S:
            return execute_load_end_get_s(state, event.rob_index, t)

        case M_Event_Name.EXECUTE_LOAD_COMPLETE:
            return execute_load_complete(state, event.rob_index)

        case M_Event_Name.COMMIT_IMMEDIATE:
            return commit_immediate(state)

        case M_Event_Name.COMMIT_ARITHMETIC:
            return commit_arithmetic(state)

        case M_Event_Name.COMMIT_BRANCH:
            return commit_branch(state)

        case M_Event_Name.COMMIT_LOAD:
            return commit_load(state)

        case M_Event_Name.COMMIT_STORE:
            return commit_store(state)
        
        case _:
            return state
            
def enabled(state: State_STT, event: M_Event, t: int) -> bool:
    match event.name:
        case M_Event_Name.EXECUTE_IMMEDIATE:
            return enabled_execute_immediate(state, event.rob_index)

        case M_Event_Name.EXECUTE_ARITHMETIC:
            return enabled_execute_arithmetic(state, event.rob_index)

        case M_Event_Name.EXECUTE_BRANCH_SUCCESS:
            return enabled_execute_branch_success(state, event.rob_index)

        case M_Event_Name.EXECUTE_BRANCH_FAIL:
            return enabled_execute_branch_fail(state, event.rob_index)

        case M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S:
            return enabled_execute_load_begin_get_s(state, event.rob_index)

        case M_Event_Name.EXECUTE_LOAD_END_GET_S:
            return enabled_execute_load_end_get_s(state, event.rob_index, t)

        case M_Event_Name.EXECUTE_LOAD_COMPLETE:
            return enabled_execute_load_complete(state, event.rob_index)

        case M_Event_Name.COMMIT_IMMEDIATE:
            return enabled_commit_immediate(state)

        case M_Event_Name.COMMIT_ARITHMETIC:
            return enabled_commit_arithmetic(state)

        case M_Event_Name.COMMIT_BRANCH:
            return enabled_commit_branch(state)

        case M_Event_Name.COMMIT_LOAD:
            return enabled_commit_load(state)

        case M_Event_Name.COMMIT_STORE:
            return enabled_commit_store(state)
        case _:
            return True # fetch events are always enabled

def fetch_event(instruction: Instruction) -> M_Event:
    event_name: M_Event_Name = None

    match instruction.name:
        case Instruction_Name.IMMED:
            event_name = M_Event_Name.FETCH_IMMEDIATE
        case Instruction_Name.OP:
            event_name = M_Event_Name.FETCH_ARITHMETIC
        case Instruction_Name.BRANCH:
            event_name = M_Event_Name.FETCH_BRANCH
        case Instruction_Name.LOAD:
            event_name = M_Event_Name.FETCH_LOAD
        case Instruction_Name.STORE:
            event_name = M_Event_Name.FETCH_STORE
    
    return M_Event(name=event_name, rob_index=None, instruction=instruction)

def execute_event(state: State_STT, rob_index: int) -> M_Event:
    _, instruction, _ = state.rob.seq[rob_index]

    event_name: M_Event_Name = None

    match instruction.name:
        case Instruction_Name.IMMED:
            event_name = M_Event_Name.EXECUTE_IMMEDIATE

        case Instruction_Name.OP:
            event_name = M_Event_Name.EXECUTE_ARITHMETIC

        case Instruction_Name.BRANCH:
            if enabled_execute_branch_success(state, rob_index):
                event_name = M_Event_Name.EXECUTE_BRANCH_SUCCESS
            elif enabled_execute_branch_fail(state, rob_index):
                event_name = M_Event_Name.EXECUTE_BRANCH_FAIL
            else:
                raise Exception("Neither of the execute events, branch success or branch failure, were enabled...")
                
        case Instruction_Name.LOAD:
            already_ended: bool = enabled_execute_load_complete(state, rob_index)
            already_begun: bool = False

            for i, ready, t_end in state.lq:
                if i == rob_index and not ready and t_end is not None:
                    already_begun = True
                    break

            if already_ended:
                event_name = M_Event_Name.EXECUTE_LOAD_COMPLETE
            elif already_begun:
                event_name = M_Event_Name.EXECUTE_LOAD_END_GET_S
            else:
                event_name = M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S

        case Instruction_Name.STORE:
            event_name = None
            

    return M_Event(name=event_name, rob_index=rob_index, instruction=None)

def commit_event(instruction: Instruction) -> M_Event:
    event_name: M_Event_Name = None

    match instruction.name:
        case Instruction_Name.IMMED:
            event_name = M_Event_Name.COMMIT_IMMEDIATE
        case Instruction_Name.OP:
            event_name = M_Event_Name.COMMIT_ARITHMETIC
        case Instruction_Name.BRANCH:
            event_name = M_Event_Name.COMMIT_BRANCH
        case Instruction_Name.LOAD:
            event_name = M_Event_Name.COMMIT_LOAD
        case Instruction_Name.STORE:
            event_name = M_Event_Name.COMMIT_STORE
    
    return M_Event(name=event_name, rob_index=None, instruction=None)

# TODO: Work out why this is necessary...
# ...for ready to be necessary (instead of just using enabled(start_state)) the following must exist:
# unenabled but ready execute events that since the beginning of the cycle have become enabled by having their non-ready conditions met
# -> non ready conditions like: storesAreReady(i), t >= t_end etc.
def ready(state: State_STT, execute_event: M_Event, t: int) -> bool:
    match execute_event.name:
        case None:
            # name is set to none when the instruction for 'execution' is a store
            return True

        case M_Event_Name.EXECUTE_IMMEDIATE:
            return True

        case M_Event_Name.EXECUTE_LOAD_END_GET_S:
            return True

        case M_Event_Name.EXECUTE_LOAD_COMPLETE:
            return True

        case M_Event_Name.EXECUTE_ARITHMETIC:
            _, dynamic_instruction, _ = state.rob.seq[execute_event.rob_index]
            operands_ready = [ True if state.ready[op] else False for op in dynamic_instruction.operands ]
            
            return operands_ready[1] and operands_ready[2]

        case M_Event_Name.EXECUTE_BRANCH_SUCCESS:
            _, dynamic_instruction, _ = state.rob.seq[execute_event.rob_index]
            operands_ready = [ True if state.ready[op] else False for op in dynamic_instruction.operands ]

            return operands_ready[0] and operands_ready[1]

        case M_Event_Name.EXECUTE_BRANCH_FAIL:
            _, dynamic_instruction, _ = state.rob.seq[execute_event.rob_index]
            operands_ready = [ True if state.ready[op] else False for op in dynamic_instruction.operands ]

            return operands_ready[0] and operands_ready[1]

        case M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S:
            _, dynamic_instruction, _ = state.rob.seq[execute_event.rob_index]
            operands_ready = [ True if state.ready[op] else False for op in dynamic_instruction.operands ]

            return operands_ready[1]

example_program: Program = random_program(20)

state_init = State_STT(0,{},{},{},{},ReorderBuffer([],0),[],[],BrPr(correct_predictions={}, program_length=len(example_program)-1),[],[],{},0)
def STT_Processor(P: Program) -> None:
    state = state_init
    t = 0
    halt =  False

    while not halt:
        state, halt = STT_Logic(P, state, t)
        t += 1

COMMIT_WIDTH: int = 2
FETCH_WIDTH: int = 2

def STT_Logic(P: Program, state: State_STT, t: int) -> Tuple[State_STT, bool]:
    state_snapshot: State_STT = deepcopy(state)

    print(f"STT_Logic for timestep {t}")

    print('\n')
    print(state)

    print("\n\t[commit]\n")
    for i in range(0, COMMIT_WIDTH):
        if state.rob.head == len(state.rob.seq):
            break
        instruction: Instruction = state.rob.seq[state.rob.head][1]
        e: M_Event = commit_event(instruction)
        if enabled(state, e, t):
            print("\t - committing " + str(instruction.name.name) + "" + str(instruction.operands))
            state = perform(state, e, t)
        else:
            break

    print('\n')
    print(state)

    print("\n\t[fetch]\n")
    for i in range(0, FETCH_WIDTH):
        if P[state.pc] is None:
            break # don't try and fetch beyond the end of the file
        instruction: Instruction = P[state.pc]
        e: M_Event = fetch_event(instruction)
        state.T = taint(state, instruction)
        print("\t - fetching the " + str(instruction.name.name) + " on line " + str(state.pc))
        state = perform(state, e, t)
        if e.name == M_Event_Name.FETCH_BRANCH and e.rob_index is None:
            break

    print('\n')
    print(state)
    
    print("\n\t[execute]\n")
    for i in range(state.rob.head, len(state_snapshot.rob.seq)): # "for i from σ.rob_head to σ_0.rob_tail − 1" # TODO: changing to just be to tail (as final instruction wasn't being executed). make sure thats correct
        instruction: Instruction = state.rob.seq[i][1]
        e: M_Event = execute_event(state, i)
        if enabled(state, e, t) and ready(state_snapshot, e, t): # and not delayed(state_snapshot, e, t): # TODO: add this once delayed is implemented
            print("\t - executing " + str(instruction.name.name) + " " + str(instruction.operands))
            state = perform(state, e, t)
            if e.name == M_Event_Name.EXECUTE_BRANCH_FAIL and e.rob_index == i:
                break

    print('\n')
    print(state)

    print("\n\t[untaint]\n")
    state = untaint(state)

    print('\n')
    print(state)

    halt: bool = P[state.pc] is None and state.rob.head == len(state.rob.seq) # "(P [σ.pc] = ⊥) ∧ (σ.rob_head = σ.rob_tail)"

    return tuple([state, halt])