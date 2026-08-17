from pathlib import Path

import torch

from icefall.lexicon import Lexicon
from word_phone_graph_compiler import WordPhoneCtcTrainingGraphCompiler


def main() -> None:
    lang_dir = Path("data/lang_phone")
    lexicon = Lexicon(lang_dir)
    compiler = WordPhoneCtcTrainingGraphCompiler(
        lang_dir=lang_dir,
        lexicon=lexicon,
        device=torch.device("cpu"),
    )

    result = compiler.expand_text("At law school, the same.")
    assert result.words == ["AT", "LAW", "SCHOOL", "THE", "SAME"]
    assert len(result.word_phone_spans) == len(result.words)
    assert result.word_phone_spans[0][0] == 0
    assert result.word_phone_spans[-1][1] == len(result.phones)
    assert result.oov_words == []
    assert result.phone_ids == compiler.texts_to_ids(["At law school, the same."])[0]
    assert all(i > 0 for i in result.phone_ids)

    repeated = compiler.texts_to_ids(["that that"])[0]
    graph = compiler.compile([repeated])
    assert graph.shape[0] == 1

    oov = compiler.expand_text("THISWORDCANNOTEXIST")
    assert oov.oov_words == ["THISWORDCANNOTEXIST"]
    assert oov.phones == ["SPN"]

    print("word-phone compiler tests passed")


if __name__ == "__main__":
    main()
