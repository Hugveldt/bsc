from dataclasses import dataclass
import enum
from typing import Dict, List, Tuple
from enum import Enum
from copy import deepcopy
from __future__ import annotations

class Instruction_Name(Enum):
    IMMED   = 0
    OP      = 1
    BRANCH  = 2
    LOAD    = 3
    STORE   = 4

# TODO: see what can be re-used between in_order and STT instructions.
# They might be identical by virtue of the register renaming abstracting the difference.
class Instruction:
    def __init__(self, name: Instruction_Name, operands: List[int]):
        self.name = name
        self.operands = operands

    def __str__(self):
        return f'Instruction(\n\tname: {self.name}\n\toperands: {self.operands}\n)'

    def perform(self):
        return

# abstract branch predictor
# TODO: decide on how to model branch prediction. Random would be easiest and shouldn't cause any problems.
class BrPr:
    def first_update(self, b: bool) -> BrPr:
        # TODO: define (STT formal pg. 7)
        # "The history register (global or local) is updated with the predicted direction b."
        return
    
    def second_update(self, pc: int, cond: bool, target: int) -> BrPr:
        # TODO: define (STT formal pg. 7)
        # "predictor is updated with both correct direction cond and correct target address target"
        return
    
    def predict(self, pc_br: int) -> Tuple[int, bool]:
        # TODO: define (STT formal pg. 7)
        # "bp.predict(pc_br) = (spc, b), for a given program counter pc_br , the function returns spc, which is the predicted target address, and b, which is a boolean value indicating whether the branch is predicted taken or not"
        return

@dataclass
class ReorderBuffer:
    seq: List[Tuple[int, Instruction, bool]]
    head: int

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

            new_taint[x_d] = tainted[x_a] or tainted[x_b]
            
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

            return not tainted[x_a] and not tainted[x_b]
        
        case Instruction_Name.LOAD:
            x_a: int = instruction.operands[1]

            return not tainted[x_a]
        
        case Instruction_Name.BRANCH:
            x_c: int = instruction.operands[0]
            x_d: int = instruction.operands[1]

            return not tainted[x_c] and not tainted[x_d]
        
        case Instruction_Name.STORE:
            x_a: int = instruction.operands[0]
            x_v: int = instruction.operands[1]

            return not tainted[x_a] and not tainted[x_v]


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
    return 0


####

# TODO: Should potentially change assertions to if statements
# TODO: Change execute events that take an `Instruction` parameter to take an rob_index instead (as in the branch events etc.)

def execute_immediate(state: State_STT, dynamic_instruction: Instruction) -> State_STT:
    new_state = deepcopy(state)

    x_d: int = dynamic_instruction.operands[0]
    k: int = dynamic_instruction.operands[1]

    assert(x_d in state.ready and state.ready[x_d] is False)

    new_state.reg[x_d] = k
    new_state.ready[x_d] = True

    return new_state

def execute_arithmetic(state: State_STT, dynamic_instruction: Instruction) -> State_STT:
    new_state = deepcopy(state)

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]
    x_b: int = dynamic_instruction.operands[2]

    assert(x_d in state.ready and state.ready[x_d] is False)

    assert(x_a in state.ready and state.ready[x_a] is True)
    assert(x_b in state.ready and state.ready[x_b] is True)

    new_state.reg[x_d] = state.reg[x_a] + state.reg[x_b] # Assuming that `op` is always addition

    new_state.ready[x_d] = True

    return new_state

def execute_branch_success(state: State_STT, rob_index: int) -> State_STT:
    new_state = deepcopy(state)

    origin_pc, dynamic_instruction, branch_prediction = state.rob.seq[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    assert(x_c in state.ready and state.ready[x_c] is True)
    assert(x_d in state.ready and state.ready[x_d] is True)

    assert(b == state.reg[x_c])
    assert(spc  == state.reg[x_d])

    new_bp = new_state.bp.second_update(origin_pc, state.reg[x_c], state.reg[x_d])
    new_state.bp = new_bp

    new_state.ckpt = [ checkpoint for checkpoint in state.ckpt if checkpoint[0] is not rob_index ] #  "ckpt | !=i"

    return new_state

def execute_branch_fail(state: State_STT, rob_index: int) -> State_STT:
    new_state = deepcopy(state)

    origin_pc, dynamic_instruction, branch_prediction = state.rob.seq[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    assert(x_c in state.ready and state.ready[x_c] is True)
    assert(x_d in state.ready and state.ready[x_d] is True)

    assert(b is not state.reg[x_c] or spc is not state.reg[x_d])

    new_pc = state.reg[x_d] if state.reg[x_c] else origin_pc + 1
    new_state.pc = new_pc
    
    rollback_ckpt: Tuple[int, int, Dict] = None

    for checkpoint in state.ckpt:
        if checkpoint[0] is rob_index and checkpoint[1] is origin_pc:
            rollback_ckpt = checkpoint
            break
    
    assert(rollback_ckpt is not None)

    new_rt = rollback_ckpt[2]
    new_state.rt = new_rt

    new_rob_seq = state.rob.seq[:rob_index]
    new_state.rob.seq = new_rob_seq

    new_lq = [ queued_load for queued_load in state.lq if queued_load[0] < rob_index ]
    new_state.lq = new_lq

    new_sq = [ queued_store for queued_store in state.sq if queued_store[0] < rob_index ]
    new_state.sq = new_sq

    new_ckpt = [ checkpoint for checkpoint in state.ckpt if checkpoint[0] < rob_index ]
    new_state.ckpt = new_ckpt

    return new_state

def execute_load_begin_get_s(state: State_STT, rob_index: int, t: int) -> State_STT:
    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    assert(x_a in state.ready and state.ready[x_a] is True)
    assert(x_d in state.ready and state.ready[x_d] is False)

    assert(StoresAreReady(state, rob_index))

    t_end = t + LoadLat(state.C, state.reg[x_a])

    lq_entry = None
    lq_index = None
    for i, entry in enumerate(state.lq):
        if entry[0] == rob_index:
            assert(entry[2] is None) # might be unnecessary
            lq_entry = deepcopy(entry)
            lq_index = i
            break

    assert(lq_entry is not None and lq_index is not None)

    lq_entry[2] = t_end # (i, False, ⊥) -> (i, False, t_end)
    new_state.lq[lq_index] = lq_entry

    new_state.C.append(state.reg[x_a])

    return new_state

def execute_load_end_get_s(state: State_STT, rob_index: int, t: int) -> State_STT:
    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    lq_entry = None
    lq_index = None
    for i, entry in enumerate(state.lq):
        if entry[0] == rob_index:
            assert(entry[2] is not None) # might be unnecessary
            lq_entry = deepcopy(entry)
            lq_index = i
            break

    assert(lq_entry is not None and lq_index is not None)

    t_end = lq_entry[2]

    assert(t_end is not None and t_end >= t)

    new_state.reg[x_d] = state.mem[state.reg[x_a]]

    lq_entry[1] = True # (i, False, t_end) -> (i, True, t_end)
    new_state.lq[lq_index] = lq_entry

    return new_state

def execute_load_complete(state: State_STT, rob_index: int) -> State_STT:
    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob.seq[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]

    lq_entry = None
    for entry in state.lq:
        if entry[0] == rob_index and entry[1] is True: # (i, True, _) ∈ lq
            lq_entry = entry
            break

    assert(lq_entry is not None)

    res: int = loadResult(state, rob_index, x_a, state.reg[x_d])

    new_state.reg[x_d] = res

    new_state.ready[x_d] = True

    return new_state