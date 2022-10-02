from dataclasses import dataclass
from enum import Enum
from typing import List, TypeAlias
import random

class Instruction_Name(Enum):
    IMMED   = 0
    OP      = 1
    BRANCH  = 2
    LOAD    = 3
    STORE   = 4

class Static_Instruction:
    def __init__(self, name: Instruction_Name, operands: List[int]):
        self.name = name
        self.operands = operands

    def __str__(self):
        return f'{self.name.name} {self.operands}'

class Dynamic_Instruction:
    def __init__(self, static_instruction: Static_Instruction, name: Instruction_Name, dynamic_operands: List[int]):
        self.static_instruction = static_instruction
        self.name = name
        self.operands = dynamic_operands

    def __str__(self):
        return f'{self.name.name} {self.operands} <{self.static_instruction.operands}>'

Program: TypeAlias = List[Static_Instruction]

def random_program(min_length: int) -> Program:
    raise Exception("random_program is likely to produce programs with out of bounds load instructions in it's current state")

    random.seed()

    program: Program = []
    registers: List[int] = []

    for instruction in range(1, min_length):
        name: Instruction_Name = random.choice(list(Instruction_Name))
        operands: List[int] = None
        match name:
            case Instruction_Name.IMMED:
                r_d, _, registers = random_register(registers)
                k = random.randint(0, max(len(program), min_length))

                operands = [r_d, k]

            case Instruction_Name.OP:
                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_d, k]))
                    
                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_a, k]))

                r_b, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_b, k]))

                operands = [r_d, r_a, r_b]

            case Instruction_Name.BRANCH:
                r_c, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_c, k]))
                    
                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_d, k]))

                operands = [r_c, r_d]

            case Instruction_Name.LOAD:
                # TODO: currently doesn't work at it produces loads from "un-stored" addresses

                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_d, k]))

                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_a, k]))

                # For now this just creates matching stores immediately before every load
                r_v, reg_is_new, registers = random_register(registers=registers, only_reuse=True)
                assert(not reg_is_new)
                program.append(Dynamic_Instruction(Instruction_Name.STORE, [r_a, r_v]))

                operands = [r_d, r_a]
            
            case Instruction_Name.STORE:
                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_a, k]))
                    
                r_v, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Dynamic_Instruction(Instruction_Name.IMMED, [r_v, k]))

                operands = [r_a, r_v]
        
        program.append(Dynamic_Instruction(name, operands))

    program.append(None)

    return program

def random_register(registers: List[int], only_reuse: bool = False) -> tuple[int, bool, List[int]]:
    possible_registers: int = len(registers) - int(only_reuse)

    reg: int = random.randint(0, possible_registers)
    
    is_new: bool = False
    if not only_reuse and reg == len(registers):
        registers.append(reg)
        is_new = True

    return tuple([reg, is_new, registers])

def print_program(program: Program) -> None:
    for i, instruction in enumerate(program):
        if instruction is None:
            return
        print(f"[{i}]\t{instruction.name.name}\t{instruction.operands}")


loop: Program = [
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

# TODO: this doesn't produce any delayed execution
speculative_load : Program = [
    Static_Instruction(Instruction_Name.IMMED,  [0, 1]),
    Static_Instruction(Instruction_Name.IMMED,  [1, 10]),
    Static_Instruction(Instruction_Name.IMMED,  [2, 999]),
    Static_Instruction(Instruction_Name.IMMED,  [3, 7]),
    Static_Instruction(Instruction_Name.STORE,  [1, 2]),
    Static_Instruction(Instruction_Name.BRANCH, [0, 3]),
    Static_Instruction(Instruction_Name.LOAD,   [4, 1]),
    Static_Instruction(Instruction_Name.LOAD,   [4, 1]),
    None
]

