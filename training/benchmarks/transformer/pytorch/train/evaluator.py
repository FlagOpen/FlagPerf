import os
import sacrebleu
import torch
from fairseq.data import data_utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.meters import StopwatchMeter
from fairseq import data, distributed_utils, utils, tokenizer


class Evaluator:

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args

    def evaluate(self, trainer):
        valid_bleu = self.score(trainer, 'test.raw.de')
        return valid_bleu

    def score(self, trainer, ref_file):
        torch.cuda.synchronize()
        args = self.args
        src_dict, tgt_dict = data_utils.load_dictionaries(args)

        model = trainer.get_model()

        # Initialize data iterator
        itr = self.dataloader.next_epoch_itr(shuffle=False)

        # Initialize generator
        gen_timer = StopwatchMeter()
        translator = SequenceGenerator(
            [model],
            tgt_dict.get_metadata(),
            maxlen=args.max_target_positions - 1,  # do not include EOS token
            beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
            use_amp=args.amp,
        )
        # Generate and compute BLEU
        predictions = []
        translations = translator.generate_batched_itr(
            itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=True, timer=gen_timer, prefix_size=args.prefix_size,
        )

        for sample_id, src_tokens, _, hypos in translations:
            # Process input and grount truth
            src_str = src_dict.string(src_tokens, args.remove_bpe)

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe
                )

                # Score only the top hypothesis
                if i == 0:
                    hypo_str = tokenizer.Tokenizer.detokenize(hypo_str, 'de')
                    predictions.append('{}\t{}'.format(sample_id, hypo_str))

        if args.distributed_world_size > 1:
            predictions = _all_gather_predictions(predictions)

        with open(os.path.join(args.data, ref_file), 'r') as reference:
            refs = [reference.readlines()]
        # reducing indexed predictions as strings is more memory efficient than reducing tuples
        predictions = [tuple(item.split('\t')) for item in predictions]
        predictions = [(int(item[0]), item[1]) for item in predictions]
        predictions.sort(key=lambda tup: tup[0])
        predictions = [hypo[1] + ('\n' if hypo[1][-1] != '\n' else '') for hypo in predictions]
        sacrebleu_score = sacrebleu.corpus_bleu(predictions, refs, lowercase=not args.test_cased_bleu).score

        if args.save_predictions:
            os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
            fname = ref_file + '.pred.update_{}'.format(trainer.get_num_updates())
            save_path = os.path.join(args.save_dir, 'predictions', fname)
            with open(save_path, 'w') as f:
                f.write(''.join(predictions))

        torch.cuda.synchronize()

        return sacrebleu_score


def _all_gather_predictions(predictions):
    ready = False
    all_ready = False
    reduced_predictions = []
    max_size = 65000
    while not all_ready:
        lst_len = len(predictions)
        size = 2000     # some extra space for python stuff
        n = 0
        while n < lst_len:
            str_len = len(predictions[n].encode('utf8')) + 8  # per string pickle overhead
            if size + str_len >= max_size:
                break
            size += str_len
            n += 1
        chunk = predictions[:n]
        predictions = predictions[n:]
        if not predictions:
            ready = True
        chunk = (ready, chunk)
        torch.cuda.synchronize()
        gathered = distributed_utils.all_gather_list(chunk, max_size=65000)
        torch.cuda.synchronize()
        reduced_predictions += [t[1] for t in gathered]
        all_ready = all([t[0] for t in gathered])

    reduced_predictions = [item for sublist in reduced_predictions for item in sublist]

    return reduced_predictions
