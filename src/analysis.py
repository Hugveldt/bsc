from copy import deepcopy
from typing import Dict, Tuple
from in_order import InO_Processor, State_InO
from STT import COMMIT_WIDTH, FETCH_WIDTH, ReorderBuffer, State_STT, M_Event, M_Event_Name, M_Event_Type, BrPr, Taint, commit_event, execute_event, fetch_event
from STT_program import Dynamic_Instruction, Program
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
        if e.name is M_Event_Name.FETCH_ARITHMETIC or e.name is M_Event_Name.FETCH_LOAD:
          x: int = e.instruction.operands[0]
          k.STT.T =  STT.taint(k.STT, e.instruction)

          if STT.tainted(k.STT, x):
            new_k.doomed[x] = True

    return new_k

STT_init: State_STT = State_STT(0,{},{},{},{},ReorderBuffer([],0),[],[],BrPr(),[],[],Taint({},{},{}),0)
InO_init: State_InO = State_InO(0,{},{})
Extended_init: State_Extended = State_Extended(STT_init, InO_init, False, Doomed({}))
def Extended_Processor(P: Program) -> None:
    k: State_Extended = Extended_init

    t = 0
    halt = False

    while not halt:
        k, halt = Extended_Logic(P, k, t)
        t += 1

    print(f"End State, k = (STT, in_order, mispredicted, doomed):\n{k.STT}\n\n{k.in_order}\n\n---- Mispredicted ---------\n\t{k.mispredicted}\n\n---- Doomed ---------------\n\t{k.doomed.doomed}\n\n---------------------------")

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

# TODO: program counter is also different as a result of incorrect branches. This must be a bug - pc should revert when squashing?
# Definition 5
def is_committed_state(in_order: State_InO, STT: State_STT, InO_init: State_InO) -> bool:
    """An in-order processor state Σ ∈ StateInO is the committed state for a STT processor state σ ∈ StateSTT, written σ ∼ Σ, if Σ is the result of executing a sequence of instructions corresponding to entries in the reorder buffer σ.rob[0], σ.rob[1], . . . , σ.rob[robhead - 1] from the initial state Σ_init."""
    init: State_InO = deepcopy(InO_init)

    committed_instructions: Program = [ instruction.static_instruction for _, instruction, _ in STT.rob.seq[:STT.rob.head] ]
    committed_instructions.append(None)

    P_result: State_InO = InO_Processor(committed_instructions, init)

    return in_order == P_result

