from datasets import load_dataset
import logging
import sys
from tqdm import tqdm

SAVE_EPOCH = 10
PATH_TO_train = 'train_str'
PATH_TO_test = 'test_str'
PATH_TO_eval = 'eval_str'

def concat_str(table: dict, question: str, answers: list, mode='train') -> str:
    prompt = ''
    prompt += 'Table: | '
    prompt += ' '.join(table['header'])
    prompt += ' | '

    prompt += ' | '.join([' '.join(row) for row in table['rows']])
    prompt += ' |'
    prompt += '\tQuestion:'
    prompt += f' {question}\t'

    prompt += 'Answers: '

    target = ' | '
    target += ' | '.join(answers)
    target += ' |'

    return prompt, target

def dataset_parser(dataset, n) -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger('parse data')


    train, test, eval = dataset['train'], dataset['test'], dataset['validation']

    eval_prompt = ''
    eval_target = ''

    logger.info('Begin to parse eval data')
    for i in tqdm(range(eval.num_rows)):
        parsed_str = concat_str(eval['table'][i], eval['question'][i], eval['answers'][i], 'eval')

        eval_prompt += parsed_str[0]
        eval_prompt += '\n'

        eval_target += parsed_str[1]
        eval_target += '\n'


        if i % SAVE_EPOCH == 0:
            with open(f'{n}_{PATH_TO_eval}_prompt.txt', 'a', encoding='utf-8') as eval_file:
                eval_file.write(eval_prompt)

            with open(f'{n}_{PATH_TO_eval}_target.txt', 'a', encoding='utf-8') as eval_file:
                eval_file.write(eval_target)

            eval_prompt = ''
            eval_target = ''

    with open(f'{n}_{PATH_TO_eval}_prompt.txt', 'a', encoding='utf-8') as eval_file:
                eval_file.write(eval_prompt)

    with open(f'{n}_{PATH_TO_eval}_target.txt', 'a', encoding='utf-8') as eval_file:
        eval_file.write(eval_target)
    logger.info('End to parse eval data')

dataset = load_dataset("wikitablequestions", "random-split-1", trust_remote_code=True)
dataset_parser(dataset, 1)

