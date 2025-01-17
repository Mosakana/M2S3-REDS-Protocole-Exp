from datasets import load_dataset
import logging
import sys
from tqdm import tqdm

SAVE_EPOCH = 10
PATH_TO_train = 'train_str.txt'
PATH_TO_test = 'test_str.txt'
PATH_TO_eval = 'eval_str.txt'

def concat_str(table: dict, question: str, answers: list, mode='train') -> str:
    target_str = ''
    target_str += 'Table: | '
    target_str += ' '.join(table['header'])
    target_str += ' | '

    target_str += ' | '.join([' '.join(row) for row in table['rows']])
    target_str += ' |'
    target_str += '\tQuestion:'
    target_str += f' {question}\t'
    if mode == 'train':
        target_str += 'Answers: '
        target_str += ' | '
        target_str += ' | '.join(answers)
        target_str += ' |'

    return target_str

def dataset_parser(dataset, n) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger('parse data')


    train, test, eval = dataset['train'], dataset['test'], dataset['validation']
    train_string = ''
    test_string = ''
    eval_string = ''

    logger.info('Begin to parse train data')
    for i in tqdm(range(train.num_rows)):
        train_string += concat_str(train['table'][i], train['question'][i], train['answers'][i])
        train_string += '\n'

        if i % SAVE_EPOCH == 0:
            with open(f'{n}_{PATH_TO_train}', 'a', encoding='utf-8') as train_file:
                train_file.write(train_string)

            train_string = ''

    with open(f'{n}_{PATH_TO_train}', 'a', encoding='utf-8') as train_file:
        train_file.write(train_string)
    logger.info('End to parse train data')
    #
    # logger.info('Begin to parse test data')
    # for i in tqdm(range(test.num_rows)):
    #     test_string += concat_str(test['table'][i], test['question'][i], test['answers'][i], 'test')
    #     test_string += '\n'
    #
    #     if i % SAVE_EPOCH == 0:
    #         with open(f'{n}_{PATH_TO_test}', 'a', encoding='utf-8') as test_file:
    #             test_file.write(test_string)
    #
    #         test_string = ''
    #
    # with open(f'{n}_{PATH_TO_test}', 'a', encoding='utf-8') as test_file:
    #     test_file.write(test_string)
    # logger.info('End to parse test data')
    #
    # logger.info('Begin to parse eval data')
    # for i in tqdm(range(eval.num_rows)):
    #     eval_string += concat_str(eval['table'][i], eval['question'][i], eval['answers'][i], 'eval')
    #     eval_string += '\n'
    #
    #     if i % SAVE_EPOCH == 0:
    #         with open(f'{n}_{PATH_TO_eval}', 'a', encoding='utf-8') as eval_file:
    #             eval_file.write(eval_string)
    #
    #         eval_string = ''
    #
    # with open(f'{n}_{PATH_TO_eval}', 'a', encoding='utf-8') as eval_file:
    #     eval_file.write(eval_string)
    # logger.info('End to parse eval data')

dataset = load_dataset("wikitablequestions", "random-split-1", trust_remote_code=True)
dataset_parser(dataset, 1)

