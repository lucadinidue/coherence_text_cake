from abc import ABC, abstractmethod
import random


class Perturbation(ABC):
    def __init__(self):
        # super().__init__()
        self.params = None
        random.seed(42)

    @abstractmethod
    def _set_parameters(self):
        pass

    def perturb_paragraph(self, paragraph):
        self._set_parameters()
        perturbed_paragraph = self._apply_perturbation(paragraph)
        return perturbed_paragraph

    @abstractmethod
    def _apply_perturbation(self, paragraph):
        pass

    @abstractmethod
    def get_perturbation_str(self):
        pass


class SwapPerturbation(Perturbation):

    def _set_parameters(self):
        self.params = random.sample(range(4), 2)

    def _apply_perturbation(self, paragraph):
        perturbed_paragraph = paragraph.sentences
        perturbed_paragraph[self.params[0]], perturbed_paragraph[self.params[1]] = perturbed_paragraph[self.params[1]], \
            perturbed_paragraph[self.params[0]]
        return perturbed_paragraph

    def get_perturbation_str(self):
        return f'swap_{self.params[0]}{self.params[1]}'


class SubPerturbation(Perturbation):

    def __init__(self, sentences_dict, skip_sentences):
        super().__init__()
        self.sentences_dict = sentences_dict
        self.skip_sentences = skip_sentences

    def _set_parameters(self):
        self.params = random.choice(range(4))

    def _apply_perturbation(self, paragraph):
        perturbed_paragraph = paragraph.sentences
        perturbed_paragraph[self.params] = self._get_sub_sentence(paragraph)
        return perturbed_paragraph

    def _get_sub_sentence(self, paragraph):
        document_id = paragraph.doc_id
        splitted_passage_id = paragraph.passage_id.split('_')
        paragraph_id = splitted_passage_id[0][1:]
        sentence_id = int(splitted_passage_id[self.params + 1]) + self.skip_sentences
        sub_sentence_id = f'd{document_id}_p{paragraph_id}_s{sentence_id}'
        sentence = self.sentences_dict[sub_sentence_id]
        return sentence

    def get_perturbation_str(self):
        return f'sub_{self.params}'


class NoPerturbation(Perturbation):

    def _set_parameters(self):
        pass

    def _apply_perturbation(self, paragraph):
        return paragraph.sentences

    def get_perturbation_str(self):
        return 'None'
