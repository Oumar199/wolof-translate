from nlp_project import *


def unify_correction(
    sentences: list,
    marks: List[Tuple] = [("«", "»", True), ("(", ")", False)],
    unified_sentences_between_pos: List[Tuple] = [(925, 930)],
):

    corrected_sentences = []

    only_end_mark = []

    only_begin_mark = []

    i = 0

    while i < len(sentences):

        for u in unified_sentences_between_pos:

            if i >= u[0] - 1 and i < u[1]:

                range_ = u[1] - u[0]

                unification = sentences[u[0] - 1]

                for j in range(u[0], u[0] + range_):

                    unification += " " + sentences[j]

                i += range_ + 1

                corrected_sentences.append(unification)

        unify_next = False

        space = " "

        if i != 0:

            for mark in marks:

                begin_mark = False

                end_mark = False

                for letter in corrected_sentences[-1]:

                    if letter == mark[1]:

                        begin_mark = False

                    elif letter == mark[0]:

                        begin_mark = True

                for letter in sentences[i]:

                    if letter == mark[1]:

                        end_mark = True

                        break

                    else:

                        break

                if end_mark and not begin_mark:

                    only_end_mark.append(sentences[i])

                elif begin_mark and not end_mark:

                    only_begin_mark.append(corrected_sentences[-1])

                if end_mark and begin_mark:

                    unify_next = True

                    space = " " if mark[2] else ""

        if unify_next:

            corrected_sentences[-1] = corrected_sentences[-1] + space + sentences[i]

        else:

            corrected_sentences.append(sentences[i])

        i += 1

    return corrected_sentences, {
        "begin_mark_only": only_begin_mark,
        "end_mark_only": only_end_mark,
    }
