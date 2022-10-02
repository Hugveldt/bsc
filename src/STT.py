from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from copy import deepcopy
from STT_program import Instruction_Name, Static_Instruction, Dynamic_Instruction, Program, random_program, print_program
import STT_program

# abstract branch predictor
class BrPr:
    def __init__(self):
        pass

    def first_update(self, b: bool) -> BrPr:
        # "The history register (global or local) is updated with the predicted direction b."
        # NOTE: currently does nothing but exists to match spec of STT paper
        return self
    
    def second_update(self, pc: int, cond: bool, target: int) -> BrPr:
        # "predictor is updated with both correct direction cond and correct target address target"
        return self
    
    def predict(self, pc_br: int) -> Tuple[int, bool]:
        # "bp.predict(pc_br) = (spc, b), for a given program counter pc_br , the function returns spc, which is the predicted target address, and b, which is a boolean value indicating whether the branch is predicted taken or not"
        # #TODO: single-direction static predictor. see wiki
        
        spc: int = pc_br + 1
        b: bool = False

        return tuple([spc, b])

class Taint:
    def __init__(self, yrot: Dict, rt_YRoT: Dict, rt_LL: Dict):
        self.yrot = yrot
        self.rt_YRoT = rt_YRoT
        self.rt_LL = rt_LL

class ReorderBuffer:
    def __init__(self, seq: List[Tuple[int, Dynamic_Instruction, bool]], head: int):
        self.seq = seq
        self.head = head

    def __getitem__(self, index: int) -> Tuple[int, Dynamic_Instruction, bool]:
        return self.seq[index]

    def __setitem__(self, index: int, entry: Tuple[int, Dynamic_Instruction, bool]) -> None:
        self.seq[index] = entry

    @property
    def tail(self):
        return len(self.seq)

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
                T: Taint,
                next_fresh: int = 0 # the lowest unused PRegID
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
        s += f'\n\tbranch predictor:\n\t\tN/A'
        s += f'\n\tbranch checkpoints: {self.ckpt}'
        s += f'\n\tcache: {self.C}'
        s += f'\n\ttaint:\n\t\tyrot: {self.T.yrot}\n\t\trt_YRoT: {self.T.rt_YRoT}\n\t\trt_LL: {self.T.rt_LL}'

        return s



def rename(state: State_STT, static_instruction: Static_Instruction) -> Tuple[Dict, Dynamic_Instruction]:
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
            dynamic_instruction: Dynamic_Instruction = Dynamic_Instruction(static_instruction, Instruction_Name.IMMED, [x_d, k])
            
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

            dynamic_instruction: Dynamic_Instruction = Dynamic_Instruction(static_instruction, Instruction_Name.OP, [x_d, x_a, x_b])

            return (new_rt, dynamic_instruction)

        case Instruction_Name.BRANCH:
            # rt is unchanged by branch
            rt_copy: Dict = deepcopy(state.rt)

            r_c: int = static_instruction.operands[0]
            r_d: int = static_instruction.operands[1]

            x_c: int = state.rt[r_c]
            x_d: int = state.rt[r_d]

            dynamic_instruction: Dynamic_Instruction = Dynamic_Instruction(static_instruction, Instruction_Name.BRANCH, [x_c, x_d])

            return (rt_copy, dynamic_instruction)

        case Instruction_Name.LOAD:
            new_rt: Dict = deepcopy(state.rt)

            r_d: int = static_instruction.operands[0]
            r_a: int = static_instruction.operands[1]

            x_d: int = fresh(state)
            new_rt[r_d] = x_d

            x_a: int = state.rt[r_a]

            dynamic_instruction: Dynamic_Instruction = Dynamic_Instruction(static_instruction, Instruction_Name.LOAD, [x_d, x_a])

            return (new_rt, dynamic_instruction)

        case Instruction_Name.STORE:
            # rt is unchanged by store
            rt_copy: Dict = deepcopy(state.rt)

            r_a: int = static_instruction.operands[0]
            r_v: int = static_instruction.operands[1]

            x_a: int = state.rt[r_a]
            x_v: int = state.rt[r_v]

            dynamic_instruction: Dynamic_Instruction = Dynamic_Instruction(static_instruction, Instruction_Name.STORE, [x_a, x_v])

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

def taint(state: State_STT, static_instruction: Static_Instruction) -> Tuple[Dict, Dict, Dict]:
    new_yrot: Dict = deepcopy(state.T.yrot)
    new_yrot[state.rob.tail] = newYRoT(state, static_instruction)

    new_rt_YRoT: Dict = deepcopy(state.T.rt_YRoT)
    new_rt_LL:   Dict = deepcopy(state.T.rt_LL)

    if static_instruction.name is Instruction_Name.LOAD or static_instruction.name is Instruction_Name.OP or static_instruction.name is Instruction_Name.IMMED:
        r_d: int = static_instruction.operands[0]

        new_rt_YRoT[r_d] = newYRoT(state, static_instruction)
        new_rt_LL[r_d]   = state.rob.tail if static_instruction.name is Instruction_Name.LOAD else None

    return tuple([new_yrot, new_rt_YRoT, new_rt_LL])

def taintingInstr(state: State_STT, r: int) -> int:
    return state.T.rt_YRoT[r] if state.T.rt_LL[r] is None else state.T.rt_LL[r]

def max_taint(t_a: Optional[int], t_b: Optional[int]) -> Optional[int]:
    if t_a is None:
        if t_b is None:
            return None
        else:
            return t_b
    else:
        if t_b is None:
            return t_a
        else:
            return max(t_a, t_b)

def newYRoT(state: State_STT, static_instruction: Static_Instruction) -> Optional[int]:
    match static_instruction.name:
        case Instruction_Name.IMMED:
            return None
        case Instruction_Name.OP:
            r_a: int = static_instruction.operands[1]
            r_b: int = static_instruction.operands[2]

            return max_taint(taintingInstr(state, r_a), taintingInstr(state, r_b))

        case Instruction_Name.LOAD:
            r_a: int = static_instruction.operands[1]

            return taintingInstr(state, r_a)

        case Instruction_Name.BRANCH:
            r_c: int = static_instruction.operands[0]
            r_d: int = static_instruction.operands[1]

            return max_taint(taintingInstr(state, r_c), taintingInstr(state, r_d))

        case Instruction_Name.STORE:
            r_a: int = static_instruction.operands[0]
            r_v: int = static_instruction.operands[1]

            return max_taint(taintingInstr(state, r_a), taintingInstr(state, r_v))

def noTaintedInputs(state: State_STT, i: int) -> bool:
    return i not in state.T.yrot or state.T.yrot[i] is None or not underSpec(state.ckpt, state.T.yrot[i])

def untaint(state: State_STT) -> State_STT:
    return state

def tainted(state: State_STT, x: int) -> bool:
    for i in range(state.rob.head, state.rob.tail-1):
        _, dynamic_instruction, _ = state.rob[i]

        if dynamic_instruction.name is Instruction_Name.LOAD and dynamic_instruction.operands[0] == x and underSpec(state.ckpt, i):
            return True
        else:
            match dynamic_instruction.name:
                case Instruction_Name.OP:
                    if dynamic_instruction.operands[0] == x and not noTaintedInputs(state, i):
                        return True
                
                case Instruction_Name.LOAD:
                    if dynamic_instruction.operands[0] == x and not noTaintedInputs(state, i):
                        return True

    return False

# μ-events

### fetch events

def fetch_immediate(state: State_STT, static_instruction: Static_Instruction) -> State_STT:
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(new_state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_arithmetic(state: State_STT, static_instruction: Dynamic_Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(new_state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_branch(state: State_STT, static_instruction: Static_Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(new_state, static_instruction)

    spc, b = state.bp.predict(state.pc)

    new_bp = new_state.bp.first_update(b)

    new_pc = spc if b else state.pc + 1

    new_ckpt = tuple([state.rob.tail, state.pc, state.rt]) # "newCkpt = (rob_tail, pc, rt)"

    new_state.pc = new_pc

    new_state.rt = new_rt

    spc_b = tuple([spc, b])
    rob_entry = tuple([state.pc, dynamic_instruction, spc_b])
    new_state.rob.seq.append(rob_entry)

    new_state.bp = new_bp

    new_state.ckpt.append(new_ckpt)

    return new_state

def fetch_load(state: State_STT, static_instruction: Static_Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(new_state, static_instruction)

    r_d: int = static_instruction.operands[0]
    x_d:int = new_rt[r_d]

    new_state.pc = state.pc + 1

    new_state.ready[x_d] = False

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    return new_state

def fetch_store(state: State_STT, static_instruction: Static_Instruction):
    new_state = deepcopy(state)

    new_rt, dynamic_instruction = rename(new_state, static_instruction)

    new_state.pc = state.pc + 1

    new_state.rt = new_rt

    rob_entry = tuple([state.pc, dynamic_instruction, None])
    new_state.rob.seq.append(rob_entry)

    new_state.sq.append(state.rob.tail) #  "sq ++ [rob_tail]"

    return new_state

### execute events

#### load helper functions

def StoresAreReady(state: State_STT, rob_i: int) -> bool:
    for rob_j in state.sq:
        if rob_j < rob_i:
            _, older_store, _ = state.rob[rob_j]

            assert(older_store.name == Instruction_Name.STORE)

            x_b: int = older_store.operands[0]

            if not state.ready[x_b]:
                return False

    return True

def CanForward(state: State_STT, rob_i: int, x_a: int, rob_j: int, x_v: int) -> bool:
    in_store_queue: bool = rob_j in state.sq
    is_younger: bool = rob_j < rob_i

    _, dynamic_instruction, _ = state.rob[rob_j]

    assert(dynamic_instruction.name == Instruction_Name.STORE)

    x_b: int = dynamic_instruction.operands[0]

    registers_ready: bool = state.ready[x_b] and state.ready[x_v]

    addresses_match: bool = state.reg[x_a] == state.reg[x_b]

    no_younger_stores: bool = True
    for rob_k in state.sq:
        if rob_j < rob_k < rob_i:
            _, younger_instruction, _ = state.rob[rob_k]

            k_v: int = younger_instruction.operands[1]

            if CanForward(state, rob_i, x_a, rob_k, k_v):
                no_younger_stores = False
                break


    return in_store_queue and is_younger and registers_ready and addresses_match and no_younger_stores

def loadResult(state: State_STT, rob_i: int, x_a: int, oldRes: int) -> int:
    for rob_j in state.sq:
        if rob_j < rob_i:
            _, dynamic_instruction, _ = state.rob[rob_j]

            x_v: int = dynamic_instruction.operands[1]

            if CanForward(state, rob_i, x_a, rob_j, x_v):
                return state.reg[x_v]
    
    return oldRes

# TODO: decide how to implement (STT formal pg. 8)
# "for a given sequence of memory addresses already accessed and the next address to access returns the number of cycles it takes to load the value by that address"
def LoadLat(cache: List[int], target_address: int) -> int:
    return 20


####

def execute_immediate(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_immediate(state, rob_index)) # result is undefined if event isn't enabled

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    k: int = dynamic_instruction.operands[1]

    new_state.reg[x_d] = k
    new_state.ready[x_d] = True

    return new_state

def enabled_execute_immediate(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, _ = state.rob[rob_index]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is True:
        return False
    
    return True

def execute_arithmetic(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_arithmetic(state, rob_index))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob[rob_index]

    x_d: int = dynamic_instruction.operands[0]
    x_a: int = dynamic_instruction.operands[1]
    x_b: int = dynamic_instruction.operands[2]

    new_state.reg[x_d] = state.reg[x_a] + state.reg[x_b] # Assuming that `op` is always addition

    new_state.ready[x_d] = True

    return new_state

def enabled_execute_arithmetic(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, _ = state.rob[rob_index]

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

    origin_pc, dynamic_instruction, _ = state.rob[rob_index]

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    new_bp = new_state.bp.second_update(origin_pc, state.reg[x_c], state.reg[x_d])
    new_state.bp = new_bp

    new_state.ckpt = [ checkpoint for checkpoint in state.ckpt if checkpoint[0] != rob_index ] #  "ckpt | != i"

    return new_state

def enabled_execute_branch_success(state: State_STT, rob_index: int) -> bool:
    _, dynamic_instruction, branch_prediction = state.rob[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    #print("branch success?")
    if x_c not in state.ready or state.ready[x_c] is False:
        #print("\tNo: comp. reg not ready")
        return False
    if x_d not in state.ready or state.ready[x_d] is False:
        #print("\tNo: dest. reg not ready")
        return False
    if b != state.reg[x_c]:
        #print("\tNo: direction prediction incorrect")
        return False
    if spc != state.reg[x_d]:
        #print("\tNo: target prediction incorrect")
        return False

    return True

def execute_branch_fail(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_branch_fail(state, rob_index))

    new_state = deepcopy(state)

    origin_pc, dynamic_instruction, _ = state.rob[rob_index]

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

    new_rob_seq = state.rob[:rob_index]
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
    origin_pc, dynamic_instruction, branch_prediction = state.rob[rob_index]
    spc, b = branch_prediction

    x_c: int = dynamic_instruction.operands[0]
    x_d: int = dynamic_instruction.operands[1]

    #print("branch failure?")
    if x_c not in state.ready or state.ready[x_c] is False:
        #print("\tNo: comp. reg not ready")
        return False
    if x_d not in state.ready or state.ready[x_d] is False:
        #print("\tNo: dest. reg not ready")
        return False
    if b == state.reg[x_c] and spc == state.reg[x_d]:
        #print("\tNo: branch prediction was correct (should be a success?)")
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

    _, dynamic_instruction, _ = state.rob[rob_index]

    x_a: int = dynamic_instruction.operands[1]

    t_end = t + LoadLat(state.C, state.reg[x_a])

    lq_entry = None
    lq_index = None
    for i, entry in enumerate(state.lq):
        #print(i)
        #print(entry)
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
    _, dynamic_instruction, _ = state.rob[rob_index]

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

    _, dynamic_instruction, _ = state.rob[rob_index]

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
    if t_end > t:
        return False

    return True

def execute_load_complete(state: State_STT, rob_index: int) -> State_STT:
    assert(enabled_execute_load_complete(state, rob_index))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob[rob_index]

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
    _, dynamic_instruction, _ = state.rob[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        #print(f"\n\tx_d of immed not ready...")
        return False

    return True

def commit_arithmetic(state: State_STT) -> State_STT:
    assert(enabled_commit_arithmetic(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    return new_state

def enabled_commit_arithmetic(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        #print(f"\n\tx_d of op not ready...")
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
        #print(f"\n\tbranch is still unresolved...")
        return False

    return True

def commit_load(state: State_STT) -> State_STT:
    assert(enabled_commit_load(state))

    new_state = deepcopy(state)

    new_state.rob.head += 1

    new_state.lq = [ entry for entry in state.lq if entry[0] != state.rob.head ]

    return new_state

def enabled_commit_load(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob[state.rob.head]

    x_d: int = dynamic_instruction.operands[0]

    if x_d not in state.ready or state.ready[x_d] is False:
        #print(f"\n\tx_d of load not ready...")
        return False

    return True

def commit_store(state: State_STT) -> State_STT:
    assert(enabled_commit_store(state))

    new_state = deepcopy(state)

    _, dynamic_instruction, _ = state.rob[state.rob.head]

    x_a: int = dynamic_instruction.operands[0]
    x_v: int = dynamic_instruction.operands[1]

    new_state.mem[state.reg[x_a]] = state.reg[x_v]

    new_state.rob.head += 1

    new_state.sq = [ entry for entry in state.sq if entry != state.rob.head ]

    new_state.C.append(state.reg[x_a])

    return new_state

def enabled_commit_store(state: State_STT) -> bool:
    _, dynamic_instruction, _ = state.rob[state.rob.head]

    x_a: int = dynamic_instruction.operands[0]
    x_v: int = dynamic_instruction.operands[1]

    if x_a not in state.ready or state.ready[x_a] is False:
        #print(f"\n\tx_a of store not ready...")
        return False
    if x_v not in state.ready or state.ready[x_v] is False:
        #print(f"\n\tx_v of store not ready...")
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

class M_Event_Type:
    Fetch_Events: List[M_Event_Name] = [
        M_Event_Name.FETCH_IMMEDIATE,
        M_Event_Name.FETCH_ARITHMETIC,
        M_Event_Name.FETCH_BRANCH,
        M_Event_Name.FETCH_LOAD,
        M_Event_Name.FETCH_STORE
    ]

    Execute_Events: List[M_Event_Name] = [
        M_Event_Name.EXECUTE_IMMEDIATE,
        M_Event_Name.EXECUTE_ARITHMETIC,
        M_Event_Name.EXECUTE_BRANCH_SUCCESS,
        M_Event_Name.EXECUTE_BRANCH_FAIL,
        M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S,
        M_Event_Name.EXECUTE_LOAD_END_GET_S,
        M_Event_Name.EXECUTE_LOAD_COMPLETE
    ]

    Commit_Events: List[M_Event_Name] = [
        M_Event_Name.COMMIT_IMMEDIATE,
        M_Event_Name.COMMIT_ARITHMETIC,
        M_Event_Name.COMMIT_BRANCH,
        M_Event_Name.COMMIT_LOAD,
        M_Event_Name.COMMIT_STORE
    ]

@dataclass
class M_Event:
    name        : M_Event_Name
    rob_index   : Optional[int]
    instruction : Optional[Dynamic_Instruction]

    def __init__(self, name: M_Event_Name, rob_index: Optional[int], instruction : Optional[Dynamic_Instruction]):
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

def fetch_event(instruction: Dynamic_Instruction) -> M_Event:
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
    _, instruction, _ = state.rob[rob_index]

    event_name: M_Event_Name = None

    match instruction.name:
        case Instruction_Name.IMMED:
            event_name = M_Event_Name.EXECUTE_IMMEDIATE

        case Instruction_Name.OP:
            event_name = M_Event_Name.EXECUTE_ARITHMETIC

        case Instruction_Name.BRANCH:
            if enabled_execute_branch_success(state, rob_index):
                event_name = M_Event_Name.EXECUTE_BRANCH_SUCCESS
            else:
                event_name = M_Event_Name.EXECUTE_BRANCH_FAIL
                
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

def commit_event(instruction: Dynamic_Instruction) -> M_Event:
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
            operands_ready = [ True if op in state.ready and state.ready[op] else False for op in execute_event.instruction.operands ]
            
            return operands_ready[1] and operands_ready[2]

        case M_Event_Name.EXECUTE_BRANCH_SUCCESS:
            operands_ready = [ True if op in state.ready and state.ready[op] else False for op in execute_event.instruction.operands ]

            return operands_ready[0] and operands_ready[1]

        case M_Event_Name.EXECUTE_BRANCH_FAIL:
            operands_ready = [ True if op in state.ready and state.ready[op] else False for op in execute_event.instruction.operands ]

            return operands_ready[0] and operands_ready[1]

        case M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S:
            operands_ready = [ True if op in state.ready and state.ready[op] else False for op in execute_event.instruction.operands ]

            return operands_ready[1]

def isExplicitBranch(e: M_Event) -> bool:
    return e.name is M_Event_Name.EXECUTE_BRANCH_SUCCESS or e.name is M_Event_Name.EXECUTE_BRANCH_FAIL

def isTransmitter(e: M_Event) -> bool:
    return e.name is M_Event_Name.EXECUTE_LOAD_BEGIN_GET_S

def delayed(state: State_STT, e: M_Event, t: int) -> bool:
    return e.rob_index is not None and (isExplicitBranch(e) or isTransmitter(e)) and not noTaintedInputs(state, e.rob_index)   

example_program: Program = STT_program.loop

state_init = State_STT(0,{},{},{},{},ReorderBuffer([],0),[],[],BrPr(),[],[],Taint({},{},{}),0)
def STT_Processor(P: Program) -> State_STT:
    state = state_init

    # DEBUG #
    #print("[Initial State]\n")
    #print(state)

    t = 0
    halt =  False

    while not halt:
        state, halt = STT_Logic(P, state, t)
        t += 1

    return state

COMMIT_WIDTH: int = 2
FETCH_WIDTH: int = 2

def STT_Logic(P: Program, state: State_STT, t: int) -> Tuple[State_STT, bool]:
    state_snapshot: State_STT = deepcopy(state)

    #print(f"STT_Logic for timestep {t}")

    #print('\n')
    #print(state)

    #print("\n\t[commit]\n")
    for i in range(1, COMMIT_WIDTH+1):
        if state.rob.tail == 0 or state.rob.head == state.rob.tail:
            break

        instruction: Dynamic_Instruction = state.rob[state.rob.head][1]
        e: M_Event = commit_event(instruction)
        if enabled(state, e, t):
            #print("\t - committing " + str(instruction.name.name) + "" + str(instruction.operands))
            state = perform(state, e, t)
        else:
            #print(f"\t...commit not enabled\n")
            break

    ##print('\n')
    ##print(state)

    #print("\n\t[fetch]\n")
    for i in range(1, FETCH_WIDTH+1):
        if P[state.pc] is None:
            break # don't try and fetch beyond the end of the file
        instruction: Dynamic_Instruction = P[state.pc]
        e: M_Event = fetch_event(instruction)

        # state.T = taint(state, instruction)
        new_yrot, new_rt_YRoT, new_rt_LL = taint(state, instruction)
        state.T = Taint(new_yrot, new_rt_YRoT, new_rt_LL)

        #print("\t - fetching the " + str(instruction.name.name) + " on line " + str(state.pc))
        state = perform(state, e, t)
        if e.name == M_Event_Name.FETCH_BRANCH and e.rob_index is None:
            break

    ##print('\n')
    ##print(state)
    
    #print("\n\t[execute]\n")
    for i in range(state.rob.head, state_snapshot.rob.tail): # "for i from σ.rob_head to σ_0.rob_tail − 1" # TODO: changing to just be to tail (as final instruction wasn't being executed). make sure thats correct
        instruction: Dynamic_Instruction = state.rob[i][1]
        e: M_Event = execute_event(state, i)
        e.instruction = instruction

        if enabled(state, e, t) and ready(state_snapshot, e, t) and not delayed(state_snapshot, e, t):
            #print("\t - executing " + str(instruction.name.name) + " " + str(instruction.operands))
            state = perform(state, e, t)
            if e.name == M_Event_Name.EXECUTE_BRANCH_FAIL:
                break

    ##print('\n')
    ##print(state)

    # #print("\n\t[untaint]\n")
    state = untaint(state)

    #print('\n')
    #print(state)

    #print(f"\n\n\t\tP[pc]: {P[state.pc]} | h={state.rob.head} vs. t={state.rob.tail}\n\n")

    halt: bool = P[state.pc] is None and state.rob.head == state.rob.tail # "(P [σ.pc] = ⊥) ∧ (σ.rob_head = σ.rob_tail)"

    return tuple([state, halt])
    