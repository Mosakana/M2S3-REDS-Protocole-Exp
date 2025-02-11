from datasets import load_dataset
import logging
import sys
from tqdm import tqdm

SAVE_EPOCH = 10
PATH_TO_train = 'train_str'
PATH_TO_test = 'test_str'
PATH_TO_eval = 'eval_str'

def concat_str(table: dict, question: str, answers: list) -> str:
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
    test_prompt = ''
    test_target = ''

    logger.info('Begin to parse test data')
    for i in tqdm(range(test.num_rows)):
        parsed_str = concat_str(test['table'][i], test['question'][i], test['answers'][i])
        test_prompt += parsed_str[0]
        test_prompt += '\n'

        test_target += parsed_str[1]
        test_target += '\n'

        if i % SAVE_EPOCH == 0:
            with open(f'{n}_{PATH_TO_test}_prompt.txt', 'a', encoding='utf-8') as test_file:
                test_file.write(test_prompt)

            with open(f'{n}_{PATH_TO_test}_target.txt', 'a', encoding='utf-8') as test_file:
                test_file.write(test_target)

            test_prompt = ''
            test_target = ''        

    with open(f'{n}_{PATH_TO_test}_prompt.txt', 'a', encoding='utf-8') as test_file:
        test_file.write(test_prompt)

    with open(f'{n}_{PATH_TO_test}_target.txt', 'a', encoding='utf-8') as test_file:
        test_file.write(test_target)
    logger.info('End to parse test data')

dataset = load_dataset("wikitablequestions", "random-split-1", trust_remote_code=True)
dataset_parser(dataset, 1)

