
import unicodedata
import numpy as np


def ctc_beam_search(
    log_probs,
    chars,
    beam_width=10,
    top_k=16
):
    """
    CTC prefix beam search.

    log_probs: NumPy array [time_steps, vocabulary_size]
    chars: character mapping from V5 checkpoint
    CTC blank token: 0
    """

    if beam_width < 1 or top_k < 1:
        raise ValueError("beam_width and top_k must be positive")

    if log_probs.ndim != 2:
        raise ValueError("Expected [time_steps, vocabulary_size]")

    if log_probs.shape[1] != len(chars) + 1:
        raise ValueError("Vocabulary does not match model output")

    NEG_INF = -float("inf")

    # prefix -> (ending_blank, ending_nonblank)
    beam = {
        (): (0.0, NEG_INF)
    }

    for timestep in log_probs:

        next_beam = {}

        candidate_ids = np.argsort(
            timestep
        )[-top_k:]

        candidate_ids = set(
            candidate_ids.tolist()
        )
        candidate_ids.add(0)

        def update(prefix, pb=None, pnb=None):

            old_pb, old_pnb = next_beam.get(
                prefix,
                (NEG_INF, NEG_INF)
            )

            if pb is not None:
                old_pb = np.logaddexp(
                    old_pb, pb
                )

            if pnb is not None:
                old_pnb = np.logaddexp(
                    old_pnb, pnb
                )

            next_beam[prefix] = (
                old_pb,
                old_pnb
            )

        for prefix, (pb, pnb) in beam.items():

            total = np.logaddexp(pb, pnb)

            update(
                prefix,
                pb=total + float(timestep[0])
            )

            for char_id in candidate_ids:

                if char_id == 0:
                    continue

                score = float(
                    timestep[char_id]
                )

                if prefix and prefix[-1] == char_id:

                    # Repeated character without blank
                    update(
                        prefix,
                        pnb=pnb + score
                    )

                    # Repeated character after blank
                    update(
                        prefix + (char_id,),
                        pnb=pb + score
                    )

                else:

                    update(
                        prefix + (char_id,),
                        pnb=total + score
                    )

        beam = dict(
            sorted(
                next_beam.items(),
                key=lambda item: np.logaddexp(
                    item[1][0],
                    item[1][1]
                ),
                reverse=True
            )[:beam_width]
        )

    best_prefix = max(
        beam,
        key=lambda prefix: np.logaddexp(
            beam[prefix][0],
            beam[prefix][1]
        )
    )

    return unicodedata.normalize(
        "NFC",
        "".join(
            chars[index - 1]
            for index in best_prefix
        )
    )
