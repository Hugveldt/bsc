from copy import deepcopy
from typing import Dict, Tuple
from in_order import InO_Processor, State_InO
from STT import COMMIT_WIDTH, FETCH_WIDTH, ReorderBuffer, State_STT, M_Event, M_Event_Name, M_Event_Type, BrPr, Taint, commit_event, execute_event, fetch_event, tainted, underSpec
from STT_program import Dynamic_Instruction, Instruction_Name, Program, print_program
import in_order
import STT
import STT_program

class Doomed:
    def __init__(self, doomed: Dict):
        self.doomed = doomed

    def __getitem__(self, PRegID: int) -> bool:
        if PRegID in self.doomed.keys():
            return self.doomed[PRegID]
        else:
            return False # doomed is False by default

    def __setitem__(self, PRegID: int, is_doomed: bool):
        self.doomed[PRegID] = is_doomed

class State_Extended:
    def __init__(self, STT: State_STT, in_order: State_InO, mispredicted: bool, doomed: Doomed):
        self.STT = STT
        self.in_order = in_order
        self.mispredicted = mispredicted
        self.doomed = doomed

def enabled(k: State_Extended, e: M_Event, t: int) -> bool:
    return STT.enabled(k.STT, e, t)

def ready(k: State_Extended, e: M_Event, t: int) -> bool:
    return STT.ready(k.STT, e, t)

def delayed(k: State_Extended, e: M_Event, t: int) -> bool:
    return STT.delayed(k.STT, e, t)

def perform(k: State_Extended, e: M_Event, t: int) -> State_Extended:
    new_k: State_Extended = deepcopy(k)

    new_k.STT = STT.perform(k.STT, e, t)

    if e.name in M_Event_Type.Fetch_Events and k.mispredicted is False:
        new_k.in_order = in_order.perform(k.in_order, e.instruction, t)

    if e.name is M_Event_Name.FETCH_BRANCH:
        taken: bool = bool(e.instruction.operands[0])

        STT_mispredicted: bool = False

        predicted_target, predicted_taken = k.STT.bp.predict(k.STT.pc)

        correct_direction: bool = predicted_taken is taken

        correctly_taken: bool = taken is True and predicted_taken is True

        correct_target: bool = predicted_target == k.in_order.pc

        if not correct_direction:
            STT_mispredicted = True
        elif correctly_taken and not correct_target:
            STT_mispredicted = True
            
        new_k.mispredicted = STT_mispredicted
    
    elif e.name is M_Event_Name.EXECUTE_BRANCH_FAIL:
        pc_br, _, _ = k.STT.rob[e.rob_index]

        STT_squashed_to_idling_point: bool = pc_br == k.in_order.pc

        if STT_squashed_to_idling_point:
            new_k.mispredicted = False
            new_k.doomed.doomed = {} # doomed now maps all PRegIDs to false

    if k.mispredicted and e.name in M_Event_Type.Fetch_Events:
        if e.name in [M_Event_Name.FETCH_ARITHMETIC, M_Event_Name.FETCH_LOAD]:
          x: int = e.instruction.operands[0]
          k.STT.T =  STT.taint(k.STT, e.instruction)

          if STT.tainted(k.STT, x):
            new_k.doomed[x] = True

    return new_k

STT_init: State_STT = State_STT(0,{},{},{},{},ReorderBuffer([],0),[],[],BrPr(),[],[],Taint({},{},{}),0)
InO_init: State_InO = State_InO(0,{},{})
Extended_init: State_Extended = State_Extended(STT_init, InO_init, False, Doomed({}))
def Extended_Processor(P: Program) -> State_Extended:
    k: State_Extended = Extended_init

    t = 0
    halt = False

    while not halt:
        k, halt = Extended_Logic(P, k, t)
        t += 1

    print(f"End State, k = (STT, in_order, mispredicted, doomed):\n{k.STT}\n\n{k.in_order}\n\n---- Mispredicted ---------\n\t{k.mispredicted}\n\n---- Doomed ---------------\n\t{k.doomed.doomed}\n\n---------------------------")

    return k

def Extended_Logic(P: Program, k: State_Extended, t: int) -> Tuple[State_Extended, bool]:
    k_snapshot: State_Extended = deepcopy(k)

    #print(k)
    #print(k.STT)
    #print(k.in_order)

    for i in range(1, COMMIT_WIDTH+1):
        if k.STT.rob.tail == 0 or k.STT.rob.head == k.STT.rob.tail:
            break

        _, instruction, _ = k.STT.rob[k.STT.rob.head]
        e: M_Event = commit_event(instruction)
        if enabled(k, e, t):
            k = perform(k, e, t)
            #print(f"\nafter commit: {k}\n{k.STT}")
        else:
            break

    for i in range(1, FETCH_WIDTH+1):
        if P[k.STT.pc] is None:
            break
        instruction = P[k.STT.pc]
        e: M_Event = fetch_event(instruction)

        new_yrot, new_rt_YRoT, new_rt_LL = STT.taint(k.STT, instruction)
        k.STT.T = Taint(new_yrot, new_rt_YRoT, new_rt_LL)

        k = perform(k, e, t)
        #print(f"\nafter fetch: {k}\n{k.STT}")
        if e.name == M_Event_Name.FETCH_BRANCH and e.rob_index is None:
            break

    for i in range(k.STT.rob.head, k.STT.rob.tail):
        _, instruction, _ = k.STT.rob[i]
        e: M_Event = execute_event(k.STT, i)
        e.instruction = instruction

        if enabled(k, e, t) and ready(k_snapshot, e, t) and not delayed(k_snapshot, e, t):
            k = perform(k, e, t)
            #print(f"\nafter execute: {k}\n{k.STT}")
            if e.name is M_Event_Name.EXECUTE_BRANCH_FAIL:
                break

    k.STT = STT.untaint(k.STT)

    halt: bool = P[k.STT.pc] is None and k.STT.rob.head == k.STT.rob.tail

    return tuple([k, halt])

example_program: Program = STT_program.loop

# TODO: The program counter won't match if there are any branch instructions as they're not represented in the ROB
# ... meaning that the committed instructions are equivalent to a program without branches
# Definition 5
def is_committed_state(in_order: State_InO, STT: State_STT, InO_init: State_InO) -> bool:
    """An in-order processor state Σ ∈ StateInO is the committed state for a STT processor state σ ∈ StateSTT, written σ ∼ Σ, if Σ is the result of executing a sequence of instructions corresponding to entries in the reorder buffer σ.rob[0], σ.rob[1], . . . , σ.rob[robhead - 1] from the initial state Σ_init."""
    init: State_InO = deepcopy(InO_init)

    committed_instructions: Program = [ instruction.static_instruction for _, instruction, _ in STT.rob.seq[:STT.rob.head] ]
    committed_instructions.append(None)

    print("\n\n -- converted committed instructions to the following program:")
    print_program(committed_instructions)
    print("\n\n")

    P_result: State_InO = InO_Processor(committed_instructions, init)

    return in_order == P_result

# Definition 6
def sim(P: Program, STT: State_STT, in_order: State_InO) -> bool:

    # (a) σ ∼ Σ
    # TODO: is the choice of initial state arbitrary? 
    def6_a: bool = is_committed_state(in_order, STT, State_InO(0,{},{}))

    # (b) if the execution of P terminated on the STT processor in σ, then so did the in-order processor in Σ, and otherwise...
    # TODO: implement this somehow
    def6_b: bool = None

    # (c) for every STT processor state σ′ and for every STT transition that advances the state from σ to σ′, there exists an in-order processor state Σ′ (possibly equal to Σ) and zero or more transitions of the in-order processor advancing Σ to Σ′ so that sim(P, σ′, Σ′).
    # TODO: implement this somehow
    def6_c: bool = None

    def6: bool = def6_a and (def6_b or def6_c)

    return def6

# Definition 7
# NOTE: could maybe compare to values of k.doomed to check correctness of "When the extended processor updates the state according to the rules above, it ensures the following property formalizing the intuitive definition of doomed:"
def doomed(k: State_Extended, x: int) -> bool:
    if not k.mispredicted or not tainted(k.STT, x):
        return False

    for i in range(k.STT.rob.head, k.STT.rob.tail):
        _, dynamic_instruction, _ = k.STT.rob[i]

        if dynamic_instruction.name in [Instruction_Name.OP, Instruction_Name.LOAD] and x == dynamic_instruction.operands[0] and underSpec(k.STT.ckpt, i):
            return True

    return False

# Definition 8
def low_equivalent(k1: State_Extended, k2: State_Extended) -> bool:
    if k1.doomed.doomed != k2.doomed.doomed:
        print("Doomed registers are not equal...")
        return False
    
    if k1.mispredicted != k2.mispredicted:
        print("Mispredicted state is not equal...")
        return False

    if k1.in_order != k2.in_order:
        print("In order states are not equal...")
        return False

    k1_non_doomed = {}
    for PRegID in k1.STT.reg.keys():
        if not k1.doomed[PRegID]:
            k1_non_doomed[PRegID] = k1.STT.reg[PRegID]

    k2_non_doomed = {}
    for PRegID in k2.STT.reg.keys():
        if not k2.doomed[PRegID]:
            k2_non_doomed[PRegID] = k2.STT.reg[PRegID]

    if k1_non_doomed == k2_non_doomed:
        print("k1 and k2 are low equivalent!")
        return True
    else:
        print("Non-doomed registers are not equal...")
        return False
