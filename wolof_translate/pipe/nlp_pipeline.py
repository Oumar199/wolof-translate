
from wolof_translate.utils.tokenize_text import tokenization
from nlp_project.processing.utils import get_n_grams, guess_limitations, wordcloud
from nlp_project import *
    

class TextPipeProcessing:
    """The pipeline is composed by (* if obligatory processing):
    - tokenize_text*
    - create_corpus*
    - create_n_grams
    - set_n_grams_as_corpus
    - reset_corpus
    - create_frequency*
    - show_frequency_out_limits
    - show_most_common_words
    - plot_frequency_histogram
    - show_n_time_frequency_words
    - delete_n_time_frequency_words
    - remove_words
    - recuperate_results*
    - add_results_to_corpus*
    - plot_wordcloud
    - some other issues...
    - use context manager to store a pipeline
    """
    pipeline = {}
    
    def __init__(self, corpus: List[str], name: str = "nlp_pipeline"):
        """Initialize main attributes

        Args:
            corpus (list): The list of documents. 
            name (str): The name of the pipeline
        """
        
        self.corpus = corpus
        
        self._corpus = None
        
        self._n_grams = None
        
        self._old_corpus = None
        
        self._grams_active = False
        
        self.bigrams = None
        
        self.trigrams = None
        
        self.name = name

    def __enter__(self):
        
        self.current_pipe = []
        
        return self
    
    def __call__(self, method: Callable, get_results: bool = True, *args, **kwargs):
    
        self.current_pipe.append({"method": method, "args": args, "kwargs": kwargs, "result": get_results})
    
    def tokenize_text(self,
                      nlp, 
                      rm_spaces: bool = True
                      ):
        """Tokenizing the text

        Args:
            nlp (_type_): The spacy model to use
            rm_spaces (bool, optional): Indicates if we want to remove the spaces. Defaults to True.

        Returns:
            List[List[str]]: List of tokens
        """
        self._nlp = nlp
        
        self._tokenizer = lambda texts: tokenization(
                      nlp,
                      texts,
                      rm_spaces
                    )
        
        self._tokens = self._tokenizer(self.corpus)
    
        return self._tokens
    
    def create_corpus(self):
        """Creating a list containing all the non distinct tokens 

        Returns:
            Tuple[list, nltk.Text]: The list of non distinct tokens and the nltk text composed of the tokens
        """
        self._corpus = []
        
        for document in tqdm(self._tokens):
            
            self._corpus.extend(document)
        
        self._corpus_text = nltk.text.Text(self._corpus)
        
        print(f"Number of words: {len(self._corpus):->16}")
        print(f"Number of unique words: {len(self._corpus_text.vocab()):->16}")
        
        return self._corpus, self._corpus_text
    
    def create_n_grams(self, n: int = 2):
        """Create n grams

        Args:
            n (int, optional): The length of a gram. Defaults to 2.

        Returns:
            Tuple[list, nltk.Text]: A list of n grams and the nltk text format of the n grams
        """
        assert n >= 2
        
        self._n_grams = []
        
        for document in tqdm(self._tokens):
            
            n_gram = get_n_grams(document, n)
            
            self._n_grams.extend(n_gram)
        
        self._n_grams_text = nltk.text.Text(self._n_grams)
        
        print(f"Number of {n} grams: {len(self._n_grams):->16}")
        print(f"Number of unique {n} grams: {len(self._n_grams_text.vocab()):->16}")

        return self._n_grams, self._n_grams_text
        
    def set_n_grams_as_corpus(self):
        """Set the n grams as the list of tokens

        Raises:
            AttributeError: The create_n_grams is not called!
        """
        self._old_corpus = self._corpus
        
        self._old_corpus_text = self._corpus_text
        
        if not self._n_grams:
            
            raise AttributeError("You didn't create the n grams with the `create_n_grams` method!")
        
        self._corpus = self._n_grams
        
        self._corpus_text = self._n_grams_text
        
        self._grams_active = True

    def reset_corpus(self):
        """Recuperating the initial corpus

        Raises:
            AttributeError: The corpus is not yet created
        """
        if not self._old_corpus:
            
            raise AttributeError("The corpus was not properly created. To create a new corpus from tokens use the `create_corpus` method!")
        
        self._corpus = self._old_corpus
        
        self._corpus_text = self._old_corpus_text
        
        self._grams_active = False
    
    def create_frequency(self):
        """Create tokens' frequencies from the list of tokens 
        """
        self._frequency = pd.DataFrame.from_dict(self._corpus_text.vocab(), 'index')
        
        self._frequency.rename({0: 'frequency'}, inplace=True, axis=1)
        
        self._frequency.reset_index(level=0, inplace=True)
        
        print(self._frequency.head())
    
    def show_frequency_out_limits(self):
        """Print the frequencies fences
        """
        px.box(data_frame=self._frequency, x="frequency", hover_data=['index', 'frequency']) 
        
        self.low, self.high = guess_limitations(self._frequency, 'frequency') 
        
        print(f"Low limit: {self.low:->16}")
        print(f"High limit: {self.high:->16}")
    
    def show_most_common_words(self, lower_bound: int = 400, n_words: int = 20):
        """Print the most common tokens (can be n grams)

        Args:
            lower_bound (int, optional): The lower bound of the frequencies. Defaults to 400.
            n_words (int, optional): The number of tokens to display. Defaults to 20.
        """
        self._freq_total = nltk.Counter(self._corpus_text.vocab())
        
        self._stopwords_common = list(zip(*self._freq_total.most_common(lower_bound)))[0]
        
        print("Most common words are:")
        print(self._stopwords_common[:20])
    
    def plot_frequency_histogram(self, bottom: int = 8):
        """Plot the histogram of the frequencies

        Args:
            bottom (int, optional): The number of the sorted frequencies to display their histograms. Defaults to 8.
        """
        f_values = self._frequency['frequency'].sort_values().unique()        
    
        bottom_ = self._frequency[self._frequency['frequency'].isin(f_values[:bottom])]
        
        fig = px.histogram(data_frame = bottom_, x = 'frequency', title=f"Frequency histogram for {bottom} frequency on the bottom", text_auto = True, color_discrete_sequence = ['indianred'])
        
        fig.show()
        
    def show_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1, n_words: int = 100):
        """Print the percentage of tokens appearing the specified number of times (frequency) in the corpus

        Args:
            n_time_freq (Union[int, list], optional): The frequency. Defaults to 1.
            n_words (int, optional): The number of words to display. Defaults to 100.
        """
        pd.options.display.max_rows = n_words
        
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        size = self._frequency[self._frequency['frequency'].isin(n_time_freq)].shape[0]
        
        n_time_frequency = self._frequency[self._frequency['frequency'].isin(n_time_freq)]
        
        print(f"Percentage of words appearing {'/'.join([str(freq) for freq in n_time_freq])} times in the dataset: {size / self._frequency.shape[0]}%")
        
        print(f"Words appearing {'/'.join([str(freq) for freq in n_time_freq])} times:")
        print(n_time_frequency.iloc[:n_words,:])
    
    def delete_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1):
        """Delete the tokens appearing a specified number of times in the corpus

        Args:
            n_time_freq (Union[int, list], optional): The number of times that that tokens appears. Defaults to 1.
        """
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        n_time_frequency = self._frequency[self._frequency['frequency'].isin(n_time_freq)]
        
        self._new_frequency = self._frequency.loc[~self._frequency['index'].isin(n_time_frequency['index'].to_list()), :]
        
        print("The new frequency data frame is stored in `_new_frequency` attribute.")
        
        print(f"The number of deleted observations: {n_time_frequency.shape[0]:->16}")
        
    def remove_words(self, words_to_remove: List[str]):
        """Remove tokens from the corpus

        Args:
            words_to_remove (List[str]): List of tokens to remove
        """
        self._new_frequency = self._new_frequency.copy()
        
        self._new_frequency.drop(index=self._new_frequency[self._new_frequency['index'].isin(words_to_remove)].index, inplace = True)
    
    def recuperate_results(self):
        """Recuperate the results as a dictionary of the tokens with their frequencies as values

        Returns:
            Tuple[nltk.FreqDist, List[tuple]]: A tuple containing the frequencies and a list of the tokens with their distinct positions in the dictionary
        """
        try:
            frequency = self._new_frequency.copy()
        except:
            frequency = self._frequency.copy()
        finally:
            print("The recuperate results method recuperates the last version of the frequency data frame as a freqDist. Make sure to add transformations before calling this method!")
        
        frequency.set_index('index', inplace = True)
        
        frequency = frequency.to_dict()
        
        frequency = frequency['frequency']
        
        self._results = nltk.FreqDist(frequency)
        
        if self._grams_active:
            
            keys = list(self._results.keys())
            
            if len(keys[0].split(" ")) == 2:
                
                self._bigrams = self._results
            
            elif len(keys[0].split(" ")) == 3:
                
                self._trigrams = self._results
        
        self._positions = {i: list(self._results.keys())[i] for i in range(len(self._results))} # positions of tokens begin at 0
        
        return self._results, self._positions
    
    def add_results_to_corpus(self):
        """Add final tokens to the corpus

        Raises:
            ValueError: Only uni grams can be added
        """
        if self._grams_active:
                
                print("You didn't reset the corpus with the `reset_corpus` method!")
        
        def clean_text(tokens: list, words: Union[nltk.FreqDist, list, set, tuple] = self._results):
            """Clean a given document by taking only words that are chosen as representative of the target

            Args:
                tokens (int): The tokens of the document
                words (Union[nltk.FreqDist, dict, list, set, tuple]): The words that we want to preserve

            Returns:
                str: The new document
            """
            
            if len(list(words.keys())[0].split(" ")) != 1:
                
                raise ValueError("Only uni grams can be provide as results to the data frame text column!")

            tokens_ = [tokens[0]]
            
            for i in range(1, len(tokens)):
                
                if tokens[i] == "-" and tokens_[-1] != "-" or tokens_[-1][-1] == "-":
                    
                    tokens_[-1] = tokens_[-1] + tokens[i] 
                
            [token for token in tokens if token in words]
            
            return " ".join(tokens_)
        
        self.corpus = list(map(clean_text), self._tokens)
        
    def plot_wordcloud(self, figsize: tuple = (8, 8), max_font_size: int = 60, max_words: int = 100, background_color = "white"):
        """Plot a wordcloud of the corpus

        Args:
            figsize (tuple, optional): The figure size with width and height. Defaults to (8, 8).
            max_font_size (int, optional): The maximum size of the font. Defaults to 60.
            max_words (int, optional): The maximum number of words on top of frequencies. Defaults to 100.
            background_color (str, optional): The background color. Defaults to "white"
        """
      
        wordcloud(" ".join(self.corpus), figsize=figsize, max_font_size=max_font_size, max_words=max_words)
            
    def predict_next_word(self, text: str):

            if self._bigrams and self._trigrams:
                
                bigram = " ".join(text.split(" ")[-2:])
                
                co_occs = []
                
                trigrams = []
                
                for trigram in self._trigrams:
                    
                    if bigram in trigram[:len(bigram)]:
                        
                        if text in set(self._bigrams.keys()):
                            
                            freq1 = self._bigrams[bigram]
                            
                            freq2 = self._trigrams[trigram]
                            
                            co_occs.append(freq2 / freq1)
                            
                            trigrams.append(trigram)

                        else:
                
                            raise KeyError(f"The bigram {text} is not identified in the registered bigrams!")
                
                try:
                
                    max_co_occ = np.array([co_occs]).argmax()
                    
                    max_trigram = trigrams[max_co_occ]
                
                    return max_trigram.split(" ")[-1], co_occs[max_co_occ]
                
                except ValueError:
                    
                    return "", None
            
            else:
                
                raise ValueError("You must create bigrams and trigrams before using them to predict the next word of your text!")
    
    def display(self, text: str, style = "dep"):
        
        # Create a container object
        doc = self._nlp(text)
        
        # Render frame with displacy
        spacy.displacy.render(doc, style=style)
    
    def execute_pipeline(self, name: str = "nlp_pipeline"):
        """Execute the pipeline

        Args:
            name (str, optional): The name of the pipeline. Defaults to "nlp_pipeline".

        Raises:
            ValueError: The pipeline name must exist before being recuperated

        Returns:
            list: The list of results
        """
        
        results = []
        
        try:
        
            pipeline = self.pipeline[name]
            
            i = 1
            
            for pipe in pipeline:
                
                args = pipe['args']
                
                kwargs = pipe['kwargs']
                
                method = pipe['method']
                
                result = pipe['result']
                
                result_ = "True" if result else "False"
                
                print(f"Method {i}: {method.__name__} -> result = {result_}\n")
                
                results_ = method(*args, **kwargs)
                
                print("\n")
                
                print("#"*100)
                
                print("\n")
                
                i += 1
                
                if result:
                    
                    results.append(results_)
            
            return results
        
        except KeyError:
            
            raise ValueError("The pipeline that you specified doesn't exist!")
    
    def __exit__(self, ctx_ept, ctx_value, ctx_tb):
        
        self.pipeline[self.name] = self.current_pipe
        
        print("You can execute the pipeline with the `pipeline_name.execute_pipeline`! The pipelines are available in the attribute `pipeline`.")
        
        return ctx_value 

