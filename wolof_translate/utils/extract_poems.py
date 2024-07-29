from typing import *


def extract_sentences(poems: List[List[str]], remove: list = ["\n"], sep: str = "_"):

    new_poems = {1: []}

    for poem in poems:

        new_poem = {1: []}

        i = 1

        j = 0

        for line in poem:

            for mark in remove:

                line = line.strip(mark).strip()

            if line == sep:

                i += 1

                j = 0

                new_poem[i] = []

            if line != "" and line != sep:

                if i > 1:

                    try:

                        line = (
                            line[0].upper() + line[1:]
                            if new_poem[i - 1][j][0].isupper()
                            else line[0].lower() + line[1:]
                        )

                    except IndexError:

                        raise IndexError(
                            "The number of lines in the different corpora are not the sames !"
                        )

                new_poem[i].append(line.strip())

                j += 1

        for key in new_poem.keys():

            if not key in new_poems:

                new_poems[key] = []

        new_poems = {k: new_poems[k] + v for k, v in new_poem.items()}

    return new_poems
