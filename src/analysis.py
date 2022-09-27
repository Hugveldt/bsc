from copy import deepcopy
from typing import Dict, Tuple
from in_order import State_InO
from STT import COMMIT_WIDTH, FETCH_WIDTH, ReorderBuffer, State_STT, M_Event, M_Event_Name, M_Event_Type, BrPr, Taint, commit_event, execute_event, fetch_event
from STT_program import Instruction, Program
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
        # TODO: work out how to define this bool... (and rename it)
        STT_mispredicted_branch: bool = True

        if STT_mispredicted_branch:
            new_k.mispredicted = True
    
    elif e.name is M_Event_Name.EXECUTE_BRANCH_FAIL:
        # TODO: work out how to define this bool... (and rename it)
        rob_index_corresponds_to_in_order_idling_point: bool = False

        if rob_index_corresponds_to_in_order_idling_point:
            new_k.mispredicted = False

    if k.mispredicted and e.name in M_Event_Type.Fetch_Events:
        if e.name is M_Event_Name.FETCH_ARITHMETIC or e.name is M_Event_Name.FETCH_LOAD:
          x: int = e.instruction.operands[0]
          k.STT.T =  STT.taint(k.STT, e.instruction)

          if STT.tainted(k.STT, x):
            new_k.doomed[x] = True
    
    elif e.name is M_Event_Name.EXECUTE_BRANCH_FAIL:
        if rob_index_corresponds_to_in_order_idling_point:
            new_k.doomed.doomed = {} # doomed now maps all PRegIDs to false

    return new_k

# TODO: define Extended_STT_Processor(...) and Extended_STT_Logic(...)
#   - Should just be a direct copy of the functions from STT.py but using the new definiton of perform and so passing an Extended State around etc.
#       -> The coordination of the STT and InO processors is handled by the new perform() function (InO executes only on non-transient fetches)
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

def Extended_Logic(P: Program, k: State_Extended, t: int) -> Tuple[State_Extended, bool]:
    k_snapshot: State_Extended = deepcopy(k)

    print(k)
    print(k.STT)
    print(k.in_order)

    for i in range(1, COMMIT_WIDTH):
        if k.STT.rob.head == k.STT.rob.tail():
            break

        _, instruction, _ = k.STT.rob.seq[k.STT.rob.head]
        e: M_Event = commit_event(instruction)
        if enabled(k, e, t):
            k = perform(k, e, t)
            print(f"\nafter commit: {k}\n{k.STT}")
        else:
            break

    for i in range(1, FETCH_WIDTH):
        if P[k.STT.pc] is None:
            break
        instruction = P[k.STT.pc]
        e: M_Event = fetch_event(instruction)

        new_yrot, new_rt_YRoT, new_rt_LL = STT.taint(k.STT, instruction)
        k.STT.T = Taint(new_yrot, new_rt_YRoT, new_rt_LL)

        k = perform(k, e, t)
        print(f"\nafter fetch: {k}\n{k.STT}")
        if e.name == M_Event_Name.FETCH_BRANCH and e.rob_index is None:
            break

    for i in range(k.STT.rob.head, k.STT.rob.tail() - 1): # TODO: spec suggests this should be `tail - 1`
        _, instruction, _ = k.STT.rob.seq[i]
        e: M_Event = execute_event(k.STT, i)

        if enabled(k, e, t) and ready(k_snapshot, e, t) and not delayed(k_snapshot, e, t):
            k = perform(k, e, t)
            print(f"\nafter execute: {k}\n{k.STT}")
            if e.name is M_Event_Name.EXECUTE_BRANCH_FAIL:
                break

    k.STT = STT.untaint(k.STT)

    halt: bool = P[k.STT.pc] is None and k.STT.rob.head == k.STT.rob.tail()

    return tuple([k, halt])

example_program: Program = STT_program.speculative_load