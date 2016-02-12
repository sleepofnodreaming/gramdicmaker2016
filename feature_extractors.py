# coding: utf-8
import math

class Thresholds(object):
    """ The class implements the previous year's threshold system.

    """
    THRESHOLD_MIN = 3
    THRESHOLD_MAX = 10
    MULTIPLIER = 1

    SPLIT_BY = u','

    @staticmethod
    def iterate_through_flex(lexeme):
        for stem in lexeme:
                for flex in lexeme[stem]:
                    yield stem, flex, lexeme[stem][flex]["freq"]

    @staticmethod
    def count_occurrences(lexeme):
        """ Given a lexeme in a json format, return the number of its word forms.

        """
        occurrences = 0
        for stem in lexeme.keys():
            for flex in lexeme[stem].keys():
                occurrences += lexeme[stem][flex]["freq"]
        return occurrences

    @staticmethod
    def count_flex(lexeme):
        """ Given a lexeme json, return the number of inflexions found with a lexeme's stems.

        """
        inflections = 0
        for stem in lexeme.keys():
            inflections += len(lexeme[stem])
        return inflections

    @staticmethod
    def count_flex_grammars_possible(lexeme):
        """ Given a lexeme json, return the number of grammatical feature sets may be postulated for lexeme's word forms.

        """
        gramFeatures = 0
        for stem in lexeme.keys():
            for flex in lexeme[stem].keys():
                gramFeatures += len(lexeme[stem][flex]["gr"])
        return gramFeatures

    @classmethod
    def list_category_values(cls, lexeme):
        categories = set()
        for stem in lexeme.keys():
            for flex in lexeme[stem].keys():
                for grammarOpt in lexeme[stem][flex]["gr"]:
                    for cat in grammarOpt.split(cls.SPLIT_BY):
                        categories.add(cat)
        return list(categories)

    @classmethod
    def binary_category_value(cls, categoryValueSet, lexeme):
        found, notFound = False, False
        for stem in lexeme.keys():
            for flex in lexeme[stem].keys():
                for grammarOpt in lexeme[stem][flex]["gr"]:
                    categories = set(grammarOpt.split(cls.SPLIT_BY))
                    if len(categories - categoryValueSet) < len(categories):
                        found = True
                    else:
                        notFound = True
        if found and notFound:
            return 2 # this means that both categories found
        return int(found or notFound)

    @classmethod
    def _get_category_distribution(cls, lexeme, category):
        catDistrib = {}
        for catValue in category:
            if catValue not in catDistrib:
                catDistrib[catValue] = 0
            for stem in lexeme:
                    for flex in lexeme[stem]:
                        for grammar in lexeme[stem][flex]["gr"]:
                            grammar = grammar.replace(u'.', cls.SPLIT_BY)
                            if catValue in grammar.split(cls.SPLIT_BY):
                                catDistrib[catValue] += lexeme[stem][flex]["freq"]
                                break
        return catDistrib

    @classmethod
    def evaluate(cls, lexeme):
        """ Count a log func value for a lexeme and tell whether it is valid or not.

        """
        occurrences = cls.count_occurrences(lexeme)
        flex = cls.count_flex(lexeme)

        # print flex, occurrences
        if flex < cls.THRESHOLD_MIN:
            return False
        if flex > cls.THRESHOLD_MAX:
            return True
        flexSupposed = cls.MULTIPLIER * math.log(occurrences, 10)
        if flex < flexSupposed:
            return False
        return True

    @classmethod
    def contains_one_flex(cls, lexeme):
        if len(list(cls.iterate_through_flex(lexeme))) == 1:
            return True
        return False


class FeatureExtractor(Thresholds):
    """ A class containing functions to extract ML features from a lexeme. Lexeme's scheme:
    {
    "eval": bool,     // is an example positive (True) or not (False)
    "lex": {       // a dic {stem (unicode) -> related info (details below)}
        "stem.": {    // a dic {flex (unicode) -> flex features}
            ".": {
                 "freq": 39, // a number of flex occurrences with this stem
                 "gr": ["sg,nom"] // flex' grammatical features possible
                 }
                 },
    "paradigm": "N-soft" // a name of a paradigm assigned
    }

    """

    LOG_BASE = 2
    CORRECTION = 0.1

    @staticmethod
    def count_flex(lexeme):
        return Thresholds.count_flex(lexeme["lex"])

    @staticmethod
    def count_occurrences(lexeme):
        return Thresholds.count_occurrences(lexeme["lex"])

    @staticmethod
    def count_flex_grammars_possible(lexeme):
        return Thresholds.count_flex_grammars_possible(lexeme["lex"])

    @staticmethod
    def iterate_through_flex(lexeme):
        return Thresholds.iterate_through_flex(lexeme['lex'])

    @classmethod
    def list_category_values(cls, lexeme):
        return Thresholds.list_category_values(lexeme['lex'])

    @classmethod
    def _get_category_distribution(cls, lexeme, category):
        return Thresholds._get_category_distribution(lexeme['lex'], category)

    @classmethod
    def proportion_flex_token(cls, lexeme):
        """ Given a lexeme, return a proportion:
        (number of different inflections / number of tokens attributed as paradigm members).
        Args:
        - lexeme.
        Return: float, 0 < result <= 1.

        """
        return float(cls.count_flex(lexeme)) / cls.count_occurrences(lexeme)

    @classmethod
    def _entropy(cls, distribution):
        """ Given a distribution as an iterable, count its entropy.
        Args:
        - distribution.
        Return: float.

        """
        h = 0
        denominator = sum(distribution)
        if denominator:
            for freq in distribution:
                probability = float(freq) / denominator
                if probability:
                    h += (probability * math.log(probability, cls.LOG_BASE))
        return -h

    @classmethod
    def entropy(cls, lexeme):
        """ Given a lexeme, count an entropy for the distribution of its inflections.
        Args:
        - lexeme.
        Return: float.

        """
        distribution = [freq for stem, flex, freq in cls.iterate_through_flex(lexeme)]
        assert distribution
        return cls._entropy(distribution)

    @classmethod
    def part_of_found_flex(cls, lexeme, pLengths):
        """ Given a lexeme, count a part of paradigm's inflections occurring in a lexeme.
        Args:
        - lexeme;
        - a dic {paradigm name -> a number of different inflections here};
        Return: float.

        """
        return float(cls.count_flex(lexeme)) / pLengths[lexeme["paradigm"]]

    @classmethod
    def part_of_found_grammars(cls, lexeme, pLengths):
        """ Given a lexeme, count a part of paradigm's grammar options occurring in a lexeme.
        Args:
        - lexeme;
        - a dic {paradigm name -> a number of different inflections here};
        Return: float.

        """
        return float(cls.count_flex_grammars_possible(lexeme)) / pLengths[lexeme["paradigm"]]

    @classmethod
    def _count_vals_per_category(cls, lexeme, catDscr):
        """ Given a lexeme and a description of all the paradigm's categories,
        count the number of different category values covered with the lexeme data.
        Args:
        - lexeme;
        - catDscr: a json containing a description of all the categories.
        Return: a list of pairs (number of values found, overall value number) for all the paradigm's categories.

        """
        assert lexeme["paradigm"] in catDscr
        intersections = []
        catVals = set(cls.list_category_values(lexeme)) # these are all the categories of a lexeme!
        for category in catDscr[lexeme["paradigm"]]:
            if len(category) > 1:
                intersection = len(category & catVals)
                pair = intersection, len(category)
            else:
                pair = cls.binary_category_value(category,  lexeme), len(category)
            intersections.append(pair)

        return intersections

    @classmethod
    def number_of_one_value_categories(cls, lexeme, catDscr):
        oneVals = [1 for intersection, gen in cls._count_vals_per_category(lexeme, catDscr) if intersection == 1]  # OMFG
        if oneVals:
            return len(oneVals)
        return 0

    @classmethod
    def entropy_to_paradigm_length(cls, lexeme, pLengths):
        h = cls.entropy(lexeme)
        # return float(h) / math.log(pLengths[lexeme["paradigm"]], cls.LOG_BASE)
        return float(h) / pLengths[lexeme["paradigm"]]

    @classmethod
    def _category_entropies(cls, lexeme, catDscr):
        """ For each paradigm's category, count an enthropy divided by a category length.
        Return a list of results.
        Args:
        - lexeme;
        - catDscr: a json containing a description of all the categories.

        """
        assert lexeme["paradigm"] in catDscr
        hs = []
        for catData in catDscr[lexeme["paradigm"]]:
                catDistrib = cls._get_category_distribution(lexeme, catData)
                # h = cls._entropy(catDistrib.values()) / math.log(len(catData), cls.LOG_BASE) # normalized!
                h = cls._entropy(catDistrib.values()) / len(catData)
                hs.append(h)

        return hs

    @classmethod
    def min_category_entropy(cls, lexeme, catDscr):
        hs = cls._category_entropies(lexeme, catDscr)
        assert hs
        return min(hs)

    @classmethod
    def average_category_entropy(cls, lexeme, categoryDescription):
        hs = cls._category_entropies(lexeme, categoryDescription)
        if hs:
            return sum(hs) / len(hs)
        return 0.0

    @classmethod
    def category_entropy_variance(cls, lexeme, catDscr):
        hs = cls._category_entropies(lexeme, catDscr)
        average = cls.average_category_entropy(lexeme, catDscr)
        return sum([(average - i)**2 for i in hs]) / len(hs)

    @classmethod
    def is_positive_example(cls, lexeme):
        """ Given a lexeme, return a boolean value showing whether this is a good one.

        """
        return lexeme["eval"]