from typing import Dict, List, Tuple, TypeAlias
from enum import Enum

class State_InO:
    def __init__(self, pc: int, mem: Dict, reg: Dict):
        self.pc = pc
        self.mem = mem
        self.reg = reg

    def __str__(self):
        return f'State_InO(\n\tpc: {self.pc}\n\tmem: {self.mem}\n\treg: {self.reg}\n)'

class Instruction_Name(Enum):
    IMMED   = 0
    OP      = 1
    BRANCH  = 2
    LOAD    = 3
    STORE   = 4

class Instruction:
    def __init__(self, name: Instruction_Name, operands: List[int]):
        self.name = name
        self.operands = operands

    def perform(self, state, t) -> State_InO:
        # TODO: Change this to be pass by value somehow so that the previous state isn't changed
        # deepcopy() ??
        match self.name:
            case Instruction_Name.IMMED:
                immed(state, self.operands[0], self.operands[1])
            case Instruction_Name.OP:
                op(state, self.operands[0], self.operands[1], self.operands[2])
            case Instruction_Name.BRANCH:
                branch(state, self.operands[0], self.operands[1])
            case Instruction_Name.LOAD:
                load(state, self.operands[0], self.operands[1])
            case Instruction_Name.STORE:
                store(state, self.operands[0], self.operands[1])

        return state

Program: TypeAlias = List[Instruction]

example_program: Program = [
    Instruction(Instruction_Name.IMMED, [0,60]),
    Instruction(Instruction_Name.IMMED, [1,61]),
    Instruction(Instruction_Name.IMMED, [2,62]),
    Instruction(Instruction_Name.IMMED, [3,63]),
    Instruction(Instruction_Name.IMMED, [4,64]),
    Instruction(Instruction_Name.IMMED, [5,65]),
    Instruction(Instruction_Name.IMMED, [6,66]),
    Instruction(Instruction_Name.IMMED, [7,67])
]

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
    state.reg[r_d] = state.mem[state.reg[r_a]]

def store(state: State_InO, r_a: int, r_v: int):
    state.pc += 1
    state.mem[state.reg[r_a]] = state.reg[r_v]

def InO_Logic(P: Program, state: State_InO, t: int) -> Tuple[State_InO, bool]:
    # TODO: Check if this bounds check is necessary/correct
    if state.pc >= len(P) or P[state.pc] == None:
        return (state, True)
    
    # These 2 lines depends on how 'P' is typed.
    # STT paper definition:
    #   e <- matching_event(P[state.pc])
    #   state' <- perform(state, e, t)
    #   return (state', false)
    state = P[state.pc].perform(state, t)
    return (state, False)

state_init = State_InO(0, {}, {})
def InO_Processor(P: Program):
    state = state_init
    t = 0
    halt = False
    while not halt:
        (state, halt) = InO_Logic(P, state, t)
        ### DEBUG
        print(state)
        ###
        t += 1