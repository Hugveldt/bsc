from copy import deepcopy
from typing import Dict
from in_order import State_InO, InO_Processor
from STT import State_STT, M_Event, M_Event_Name, M_Event_Type, fetch_event, STT_Processor
import in_order
import STT

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