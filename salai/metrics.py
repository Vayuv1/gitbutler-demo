# asr_finetune/common/metrics.py
from typing import Optional, List

import torch
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode


def _indices_to_text(vocab: List[str], idx_seqs: List[List[int]]) -> List[str]:
    return ["".join(vocab[i] for i in seq) for seq in idx_seqs]


def compute_wer(brain: sb.Brain, dataset, label_encoder, beam_size: int = 1) -> Optional[float]:
    device = next(brain.modules["encoder"].parameters()).device
    loader = sb.dataio.dataloader.SaveableDataLoader(dataset, batch_size=1)
    er = sb.utils.metric_stats.ErrorRateStats()
    vocab = label_encoder.ind2lab
    blank_id = label_encoder.get_blank_index()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            wavs, wav_lens = batch.sig
            feats = brain.hparams.compute_features(wavs)
            feats = brain.hparams.normalize(feats, wav_lens)
            enc_out = brain.modules.encoder(feats)
            if isinstance(enc_out, tuple):
                enc_out = enc_out[0]
            logits = brain.modules.ctc_lin(enc_out)
            p_ctc = brain.hparams.log_softmax(logits)

            if beam_size and beam_size > 1:
                beams_per_utt = brain.hparams.ctc_beam_searcher(p_ctc, wav_lens)
                hyps = [beams[0].text for beams in beams_per_utt]
            else:
                idxs = ctc_greedy_decode(p_ctc, wav_lens, blank_id=blank_id)
                hyps = _indices_to_text(vocab, idxs)

            # references
            refs = []
            for t, l in zip(batch.tokens, batch.tokens_lens):
                ids = t[: int(l)].tolist()
                refs.append("".join(vocab[i] for i in ids))

            er.append(batch.id, hyps, refs)
    return er.summarize()
