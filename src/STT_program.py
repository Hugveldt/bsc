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

class Instruction:
    def __init__(self, name: Instruction_Name, operands: List[int]):
        self.name = name
        self.operands = operands

    def __str__(self):
        return f'{self.name.name} {self.operands}'

Program: TypeAlias = List[Instruction]

def random_program(min_length: int) -> Program:
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
                    program.append(Instruction(Instruction_Name.IMMED, [r_d, k]))
                    
                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_a, k]))

                r_b, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_b, k]))

                operands = [r_d, r_a, r_b]

            case Instruction_Name.BRANCH:
                r_c, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_c, k]))
                    
                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_d, k]))

                operands = [r_c, r_d]

            case Instruction_Name.LOAD:
                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_d, k]))

                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_a, k]))

                # For now this just creates matching stores immediately before every load
                r_v, reg_is_new, registers = random_register(registers=registers, only_reuse=True)
                assert(not reg_is_new)
                program.append(Instruction(Instruction_Name.STORE, [r_a, r_v]))

                operands = [r_d, r_a]
            
            case Instruction_Name.STORE:
                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_a, k]))
                    
                r_v, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_v, k]))

                operands = [r_a, r_v]
        
        program.append(Instruction(name, operands))

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
    Instruction(Instruction_Name.IMMED,  [0, 10]),
    Instruction(Instruction_Name.IMMED,  [1, -1]),
    Instruction(Instruction_Name.IMMED,  [2, 999]),
    Instruction(Instruction_Name.IMMED,  [4, 7]),
    Instruction(Instruction_Name.IMMED,  [5, 9]),
    Instruction(Instruction_Name.IMMED,  [6, 12]),
    Instruction(Instruction_Name.IMMED,  [7, 1]),
    Instruction(Instruction_Name.BRANCH, [0, 5]),
    Instruction(Instruction_Name.BRANCH, [7,  6]),
    Instruction(Instruction_Name.STORE,  [0, 2]),
    Instruction(Instruction_Name.OP,     [0, 0, 1]),
    Instruction(Instruction_Name.BRANCH, [7,  4]),
    None
]

