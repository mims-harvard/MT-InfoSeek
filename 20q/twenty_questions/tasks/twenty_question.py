from twenty_questions.tasks.prompts import twenty_question


class Q20Task:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = args.max_turn
        self.prompts = twenty_question
        self.set = []
        self.data = self.load_dataset(args.dataset)

    def load_dataset(self, name):
        from twenty_questions.data.data_20q import BIG_BENCH_CONCEPT, COMMON, THING200

        datasets = {
            "bigbench": BIG_BENCH_CONCEPT,
            "common": COMMON,
            "thing": THING200,
        }
        if name not in datasets:
            raise NotImplementedError

        self.set = datasets[name]
        return [{"target": x} for x in self.set]

