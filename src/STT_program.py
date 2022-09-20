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
                k = random.randint(-1000, 1000)

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
                # TODO: need to make sure there is an accompanying store beforehand. Check if one exists otherwise create one?

                r_d, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_d, k]))
                    
                r_a, reg_is_new, registers = random_register(registers)
                if reg_is_new:
                    k = random.randint(0, len(registers)-1)
                    program.append(Instruction(Instruction_Name.IMMED, [r_a, k]))

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

def random_register(registers: List[int]) -> tuple[int, bool, List[int]]:
    reg: int = random.randint(0, len(registers))
    
    is_new: bool = False
    if reg == len(registers):
        registers.append(reg)
        is_new = True

    return tuple([reg, is_new, registers])

def print_program(program: Program) -> None:
    for i, instruction in enumerate(program):
        if instruction is None:
            return
        print(f"[{i}]\t{instruction.name.name}\t{instruction.operands}")
