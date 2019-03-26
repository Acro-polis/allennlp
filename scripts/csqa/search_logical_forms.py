from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.semantic_parsing.csqa.csqa import CSQADatasetReader
from allennlp.state_machines.pruned_breadth_first_search import PrunedBreadthFirstSearch

import sys
import os
import time
import argparse
import pickle
from collections import Counter
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(os.path.abspath(''), os.pardir)))))


def search(args):
    params = {'lazy': True,
              'kg_path': args.wikidata_directory,
              'kg_type_path': args.wikidata_type_directory,
              'entity_id2string_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
              'predicate_id2string_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'}

    reader = CSQADatasetReader.from_params(Params(params))
    csqa_directory = f'{AllenNlpTestCase.PROJECT_ROOT}/{args.csqa_directory}'
    dataset = reader.read(csqa_directory)

    logical_form_result_dict = pickle.load(open(args.initial_dict, "rb")) if args.initial_dict else {}
    comp_quan_types = ["Quantitative Reasoning (All)", "Comparative Reasoning (All)"]
    start_time = time.time()
    question_type_counter = Counter()
    stop_after_n_found = 1

    for n, instance in enumerate(tqdm(dataset, total=args.instance_limit)):
        if n == args.instance_limit:
            break

        question = [tok.text for tok in instance['question'].tokens]
        language = instance['world'].metadata
        expected_result = instance['expected_result'].metadata
        qa_id = instance['qa_id'].metadata
        question_type = instance['question_type'].metadata
        question_description = instance['question_description'].metadata
        question_type_entities = language.kg_context.question_type_entities
        searcher = PrunedBreadthFirstSearch(language)

        time_limit = 20
        question_type_counter[question_type] += 1

        if qa_id in logical_form_result_dict and len(logical_form_result_dict[qa_id]) >= stop_after_n_found:
            continue

        if question_type in comp_quan_types:
            time_limit = 30
            if question_description.endswith("Mult. entity type"):
                continue

        result_action_sequences = searcher.search(question, question_type, question_description,
                                                  question_type_entities, expected_result,
                                                  verbose=False,
                                                  max_depth=20,
                                                  max_time=time_limit,
                                                  stop_after_n_found=stop_after_n_found)
        if len(result_action_sequences) != 0:
            logical_form_result_dict[qa_id] = result_action_sequences

    with open(csqa_directory + "_logical_forms.p", 'wb') as file:
        pickle.dump(logical_form_result_dict, file)

    with open(csqa_directory + "_logical_forms_log", 'w') as file:
        n_found = len(list(logical_form_result_dict.keys()))
        percentage = n_found / args.instance_limit * 100
        file.write("{} qa turns in total, found {} ({:.2f}%)".format(args.instance_limit, n_found, percentage))
        file.write("\n")
        file.write("took {} seconds".format(time.time() - start_time))
        file.write("\n\n")
        file.write("\n".join([str(k) + " : " + str(v) for k, v in question_type_counter.items()]))
        file.write("\n\n")
        file.write("\n".join([str(k) + " : " + str(v) for k, v in vars(args).items()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikidata_directory", type=str, help="path to graph wikidata")
    parser.add_argument("--wikidata_type_directory", type=str, help="path to wikidata type graph")
    parser.add_argument("--csqa_directory", type=str, help="path to csqa directory")
    parser.add_argument("--instance_limit", type=int, help="path to initial dict")
    parser.add_argument("--initial_dict", type=str, default=None, help="path to initial dict")
    args = parser.parse_args()
    search(args)
