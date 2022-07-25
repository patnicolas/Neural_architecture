from unittest import TestCase

import pandas as pd
from wordembedding.context import Context


class TestContext(TestCase):
    def test_apply(self):
        contents = [
            'hello|why|not|Paris|is|clear',
            'not|beauty|march|april|may|borrowed|next'
        ]
        is_symmetric = True
        context_size = 3
        try:
            bag_of_words_df = pd.DataFrame(contents)
            context = Context(context_size, is_symmetric)
            parsed_df = context.apply_df(bag_of_words_df)
            values = parsed_df.values
            for pairs_list in values:
                for context, target in pairs_list:
                    print(f'Ctx: {context}, tgt: {target}')
        except Exception as e:
            self.fail(str(e))

    def test_apply_single(self):
        content = 'not|beauty|march|april|may|borrowed|next'
        is_symmetric = True
        context_size = 3
        try:
            context = Context(context_size, is_symmetric)
            values = context.apply(content)
            print(str(values))
        except Exception as e:
            self.fail(str(e))


