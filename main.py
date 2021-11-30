import numpy as np
import pandas as pd

from pathlib import Path

import QuestionAnswers.part1 as part1
import QuestionAnswers.part2 as part2

pd.options.mode.chained_assignment = None  # default='warn'


def main():
    part1.q2()
    part1.q3()
    part2.q2a()
    part2.q2b()
    part2.q2c()
    part2.q2e()


if __name__ == '__main__':
    main()
