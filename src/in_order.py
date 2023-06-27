from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple
from STT_program import Instruction_Name, Program, Static_Instruction

class State_InO:
    def __init__(self, pc: int, mem: Dict, reg: Dict):
        self.pc = pc
        self.mem = mem
        self.reg = reg

    def __str__(self):
        return f'---- State_InO ------------\n\tpc: {self.pc}\n\tmem: {self.mem}\n\treg: {self.reg}\n)'

    def __ne__(self: State_InO, other: State_InO):
        equal_pc : bool = self.pc  == other.pc
        equal_mem: bool = self.mem == other.mem
        equal_reg: bool = self.reg == other.reg

        return not equal_pc or not equal_mem or not equal_reg


example_program: Program = [
    Static_Instruction(Instruction_Name.IMMED,  [0, 10]),
    Static_Instruction(Instruction_Name.IMMED,  [1, -1]),
    Static_Instruction(Instruction_Name.IMMED,  [2, 999]),
    Static_Instruction(Instruction_Name.IMMED,  [4, 7]),
    Static_Instruction(Instruction_Name.IMMED,  [5, 9]),
    Static_Instruction(Instruction_Name.IMMED,  [6, 12]),
    Static_Instruction(Instruction_Name.IMMED,  [7, 1]),
    Static_Instruction(Instruction_Name.BRANCH, [0, 5]),
    Static_Instruction(Instruction_Name.BRANCH, [7,  6]),
    Static_Instruction(Instruction_Name.STORE,  [0, 2]),
    Static_Instruction(Instruction_Name.OP,     [0, 0, 1]),
    Static_Instruction(Instruction_Name.BRANCH, [7,  4]),
    None
]

# TODO: refactor these operations to the same format as in STT (deepcopy state and then return new state) -> State_InO:
def immed(state: State_InO, r_d: int, k: int):
    state.pc += 1
    state.reg[r_d] = k

def op(state: State_InO, r_d: int, r_a: int, r_b: int):
    state.pc += 1
    # TODO: Confirm that the choice of arithmetic operation is arbitrary. '+' in this case.
    state.reg[r_d] = state.reg[r_a] + state.reg[r_b]

def branch(state: State_InO, r_c: int, r_d: int):
    state.pc = state.reg[r_d] if state.reg[r_c] else state.pc + 1

def load(state: State_InO, r_d: int, r_a: int):
    state.pc += 1
    state.reg[r_d] = state.mem[state.reg[r_a]] if state.reg[r_a] in state.mem else 0

def store(state: State_InO, r_a: int, r_v: int):
    state.pc += 1
    state.mem[state.reg[r_a]] = state.reg[r_v]

def matching_event(instruction: Static_Instruction) -> Static_Instruction:
    return instruction

def perform(state: State_InO, e: Static_Instruction, t: int) -> State_InO:
    new_state: State_InO = deepcopy(state)

    match e.name:
        case Instruction_Name.IMMED:
            immed(new_state, e.operands[0], e.operands[1])
        case Instruction_Name.OP:
            op(new_state, e.operands[0], e.operands[1], e.operands[2])
        case Instruction_Name.BRANCH:
            branch(new_state, e.operands[0], e.operands[1])
        case Instruction_Name.LOAD:
            load(new_state, e.operands[0], e.operands[1])
        case Instruction_Name.STORE:
            store(new_state, e.operands[0], e.operands[1])

    return new_state

def InO_Logic(P: Program, state: State_InO, t: int) -> Tuple[State_InO, bool]:
    if P[state.pc] == None:
        return (state, True)

    e = matching_event(P[state.pc])
    new_state = perform(state, e, t)
    return (new_state, False)

def InO_Processor(P: Program, state_init: State_InO) -> State_InO:
    state = state_init
    t = 0
    halt = False
    while not halt:
        (state, halt) = InO_Logic(P, state, t)
        ### DEBUG
        print(state)
        ###
        t += 1
    
    return state