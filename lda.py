
class LDA:
    def __init__(self, data_file_path, num_topics, num_iterations, alpha, beta, save_data_iteration, initialize_strategy, read_from_file=True, data=[]):
        '''
        initilizes the LDA object and parameters of the class
        parameters:
            data_file_path : str : file path to the corpus
            num_topics : int : number of topics to work on
            num_iterations :int : number of iterations to work on
            alpha : float: value for alpha parameter
            beta : float : value of beta parameter
            save_data_iteration : int : Number of iterations after which the data will be saved. Used to keep track of the progress. -1 means dont save the data intermediately
            initialize_strategy : str : options [random or uniform] : How to initialize topics for words. 2 options: random or uniform. uniform means same topic for all words in a document
            read_from_file : bool : if True, then read the corpus from the file. if False, then read the corpus from the data parameter
            data : [str] : list of documents. each item in the list is a string representing the document text
        return:
            None        
        '''
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.save_data_iteration = save_data_iteration
        self.initialize_strategy = initialize_strategy
        self.current_iteration = 0  #not properly implemented at the moment

        if read_from_file:
            self.load_data(data_file_path)
        else:
            self.load_data(data, read_from_file=False)

        self.initialize()

        print("Parameters :")
        print(self)

    def __str__(self) -> str:
        ret = "len(self.vocab) : " + str(len(self.vocab)) + "\n"
        ret = ret + "len(self.data) : " + str(len(self.data)) + "\n"
        ret = ret + "self.num_documents : " + str(self.num_documents) + "\n"
        ret = ret + "num_topics : " +  str(self.num_topics) + "\n"
        ret = ret + "num_iterations : " + str(self.num_iterations) + "\n"
        ret = ret + "alpha : " + str(self.alpha) + "\n"
        ret = ret + "beta : " + str(self.beta) + "\n"

        return ret

    def load_data(self, my_arg="", read_from_file=True):
        '''
        Loads the corpus into self.data and also create self.vocab*. details about vocab* variables in get_vocab_words() function. self.data is a list of strings. each string representing a document
        parameters:
            if read_from_file=True :
                my_arg = path to corpus file
            if read_from_file=False :
                my_arg = list of documents. each item in the list is a string representing the document text 
            data_file_path : str : path to corpus file
            read_from_file : bool : if True, then read the corpus from the file. if False, then read the corpus from the data parameter
            data : [str] : list of documents. each item in the list is a string representing the document text
        return:
            None
        '''
        
        if read_from_file:
            data_file_path = my_arg
    
            f = open(data_file_path,"r")
            self.data = []  #list of strings. each string representing a document text
            for index,each_line in enumerate(f):
                if index==0:
                    self.num_documents = int(each_line)                 #assuming the first line is the number of documents
                else:
                    self.data.append(each_line.rstrip())

        else: 
            self.data = my_arg 
            #clean the data
            self.data = [self.clean_text(each_doc) for each_doc in self.data]
            self.num_documents = len(self.data)
        
        # Extra comments: 
        # creating dictionary of vocab with their index as the value. this way while fetching the index it will be O(1) instead of O(n) everytime. memory vs time tradeoff but I have memory
        # this reduced the time of the intialization of matrices from 5 mins(if self.vocab is a list...15 min if self.vocab is a set) to 3 secs. we need the vocabulary index of a word in topic_word_counts matrix
        # the only reason to keep the list is because we need vocab index in top_words() function and i am prefering to spend memory and save time so instead of wasting to check the index in the dictionary, i am keeping a list.
        # vocab_with_count is kept for pyLDAvis.
        self.vocab, self.vocab_list, self.vocab_with_count = self.get_vocab_words(self.data)

    def clean_text(self, txt):
        '''
        cleans the corpus. removes the punctuations and numbers. converts all the words to lower case.
        '''
        txt = txt.lower()
        #replace "\t and \n" with space
        txt = txt.replace("\t", " ")
        txt = txt.replace("\n", " ")
        txt = txt.translate(str.maketrans('', '', string.punctuation))
        #remove all extra spaces
        txt = " ".join(txt.split())
        return txt

    def get_vocab_words(self, data_list):
        '''
        used to create vocabulary.
        params:
            data_list : [str] : list of documents, each item representing the document text
        return:
            vocab_list ( list(vocab_dict.keys()) ) : [str] : set of unique words from the corpus
            vocab_dict : {keys: unique words | values: index of the number / ith unique word that we have seen} : created this dictionary to make the gibbs sampling faster because we often need to know the index of a word. and having words as keys and (list) index as respective value, decreases the time complexity of that operation from O(N) to O(1) 
            vocab_with_count : [int] : created to help in pyLDAvis visualization. each value represents the number of times this ith word has appeared. word itself can be fetched from the vocab_list as this list only contains the count of each word appearing in corpus  
        '''
        vocab_dict = {}
        dictionary_counter = 0
        vocab_with_count = []                                                   #this will keep the number of times each word has appeared . used for pyLDA vis
        for string in data_list:
            for word in string.split():
                if word not in vocab_dict.keys():
                    vocab_dict[word] = dictionary_counter
                    dictionary_counter += 1
                    vocab_with_count.append(0)
                else:
                    vocab_with_count[vocab_dict[word]] += 1
                    
        return vocab_dict, list(vocab_dict.keys()), vocab_with_count            #we used set to get the unique but later we need to get index and sets dont have indexes so thats why cnverting to list so we dont have to convert to list each time we wanna check the index

    def initialize(self):
        """
        Initialize topic-word and document-topic counts. Do random topic assignments for each word in each document.       
        Parameters: 
            none because we use class variables        
        Returns: 
            none because we use class variables 
        """

        print("Initializing the matrices...")
        print("")
        #initialize the document-topic counts matrix with zeros. each item i(document) has j(topic) entries and each j entry represent how many times did j appear in i
        self.document_topic_counts = np.zeros((self.num_documents, self.num_topics),dtype=float) # shape: [num_documents, num_topics]
        self.document_topic_dist = np.full((self.num_documents, self.num_topics),dtype=float, fill_value=-1) # shape: [num_documents, num_topics]

        #initialize the topic-word counts matrix with zeros.  each item i(topic) has j(word) entries and each j entry represent how many times did j appear in i
        self.topic_word_counts = np.zeros((self.num_topics, len(self.vocab)),dtype=float)     # shape: [num_topics, number of unique words in corpus]
        self.topic_word_dist = np.full((self.num_topics, len(self.vocab)),dtype=float, fill_value=-1)     # shape: [num_topics, number of unique words in corpus]


        #initialize the topic-word counts array with zeros. each item i (topic) represents how many times did topic i appear in the corpus 
        self.topic_counts = np.zeros(self.num_topics,dtype=float) # shape : [num_topics]
        
        #initialize the topic assignments matrix with zeros. each item i(document) has j(word) entries and each j entry represent how many times did j appear in i
        self.topic_assignments = [[] for _ in range(self.num_documents)] # shape: [num_documents, number of words in document] , shape will be this after the initialization: Not a square matrix : count of j will be different for each i, depending on the number of words in i document

        
        for document_number in tqdm(range(self.num_documents)):         #for all documents in corpus
            
            document = self.data[document_number]                       #get the current document
            
            document_words = document.split(" ")                        #split the document into words

            document_word_assignments = []                              # shape : [number of words in current document] keeping a local list which we will append to topic assignments at the end
            if self.initialize_strategy == "uniform":               #if unioform topics are to be assigned then just get a random one and will assign it to every word
                document__words_uniform_topic = np.random.randint(0,self.num_topics)       #randomly chose a topic

            for word_index, word in enumerate(document_words):          #for all words of current document
                if self.initialize_strategy == "uniform":
                    word_topic = document__words_uniform_topic
                else:
                    word_topic = np.random.randint(0,self.num_topics)       #randomly chose a topic
                try:
                    word_index_in_vocab = self.vocab[word]                  # the index of the current word in the vocabulary. needed for topic_word_counts matrix 
                except:
                    print("document_number : ", document_number)
                    print("word_index not in vocab : ", word_index)
                    print("word not in vocab : ", word)

                document_word_assignments.append(word_topic)

                self.topic_word_counts[word_topic][word_index_in_vocab] += 1
                self.document_topic_counts[document_number][word_topic] += 1
                self.topic_counts[word_topic] += 1
                
            self.topic_assignments[document_number] = document_word_assignments    # shape: [num_documents, number of words in document]

    def gibbs_sampling(self, previousIterationsDone = -1):
        """
        Perform Gibbs sampling to infer the topic assignments for each word in each document.
        
        Parameters:
            None
        
        Returns:
            None
        """

        print("Performing Gibbs Sampling...")

        #iterate over the number of iterations

        for iteration in range(self.num_iterations):
            print("Iteration : ", iteration)
            print("Previous Iterations Done : ", previousIterationsDone)
            self.current_iteration = iteration + previousIterationsDone
            #iterate over each document
            print("Iteration : ", self.current_iteration)

            if iteration != 0 and self.save_data_iteration!=-1 and iteration % self.save_data_iteration == 0 :
                #dist is assigned here so if we want to load and visualize using the intermediate lda model, we can. otherwise they are empty because they are assigned at the very end and hence we cant visualize if we dont do this. 
                #similar to topic word list matrix but intead of direct counts its a distribution 
                self.topic_word_dist = self.topic_word_counts / self.topic_word_counts.sum(axis=1)[:, None]  # shape (20, 46517)
                #similar to document_topic_counts matrix but intead of direct counts its a distribution 
                self.document_topic_dist = self.document_topic_counts / self.document_topic_counts.sum(axis=1)[:, None]  # shape (2000, 20)
                if previousIterationsDone!=-1:
                    self.save_variables(current_iteration=iteration+previousIterationsDone)                       #Intermediate save during training 
                else:
                    self.save_variables(current_iteration=iteration)                       #Intermediate save during training

            for document_number in tqdm(range(self.num_documents)):
            # for document_number in range(self.num_documents):
                
                #get the current document
                document = self.data[document_number]
                #split the document into words
                document_words = document.split(" ")
                
                
                for word_index, word in enumerate(document_words):
                    #The code below does the work of both 1) loop over words in documents and 2) loop over number of topics for each word of the document. I should clarify that i didnt write this code directly, as i wrote each loop separately first and then reduced the loops one by one
                    word_topic = self.topic_assignments[document_number][word_index]
                    word_index_in_vocab = self.vocab[word]

                    self.topic_word_counts[word_topic][word_index_in_vocab] -= 1
                    self.document_topic_counts[document_number][word_topic] -= 1
                    self.topic_counts[word_topic] -= 1

                    #gibbs sampling - main formula start. 
                    # calculated for current word and all the possible topics
                    rhs_denominator = (len(document_words) - 1) + (self.num_topics * self.alpha)  #scallar
                    lhs_numerator = self.topic_word_counts[:, word_index_in_vocab] + self.beta  # shape : [num_topics]
                    lhs_denominator = self.topic_counts + (len(self.vocab) * self.beta) # shape : [num_topics]
                    rhs_numerator = self.document_topic_counts[document_number, :] + self.alpha # shape : [num_topics]

                    lhs = lhs_numerator / lhs_denominator           # shape : [num_topics]
                    rhs = rhs_numerator / rhs_denominator           # shape : [num_topics]
                    word_topic_probs = np.multiply(lhs, rhs)        # shape : [num_topics]
                    #gibbs sampling - main formula end
                    #normalize
                    topic_probs = np.array(word_topic_probs) / sum(word_topic_probs)        # shape : [num_topics]

                    new_topic = np.random.choice(self.num_topics, p=topic_probs)        # scallar

                    self.topic_assignments[document_number][word_index] = new_topic

                    self.topic_word_counts[new_topic][word_index_in_vocab] += 1
                    self.document_topic_counts[document_number][new_topic] += 1
                    self.topic_counts[new_topic] += 1

    def fit(self, previousIterationsDone=-1):

        print("Fitting the LDA model...")

        self.gibbs_sampling(previousIterationsDone= previousIterationsDone)
        #similar to topic word list matrix but intead of direct counts its a distribution 
        self.topic_word_dist = self.topic_word_counts / self.topic_word_counts.sum(axis=1)[:, None]  # shape (20, 46517)
        #similar to document_topic_counts matrix but intead of direct counts its a distribution 
        self.document_topic_dist = self.document_topic_counts / self.document_topic_counts.sum(axis=1)[:, None]  # shape (2000, 20)

    def top_words(self, n):
        top_words_list = []
        for topic_index, topic in enumerate(range(self.num_topics)):
            top_words = np.argsort(self.topic_word_dist[topic])[::-1][:n]
            top_words = [self.vocab_list[i] for i in top_words]
            print("The most probable words for topic #" + str(topic_index) + " are : " ,end="" )
            print(top_words)
            top_words_list.append(top_words)
        return top_words_list

    def most_probable_topic(self):
        topics = []
        for document_number in range(self.num_documents):
            topic = self.document_topic_dist[document_number].argmax()
            topics.append(topic)
            print("The most probable topic for document #", str(document_number+1), " is --> ", topic)  
        return topics

    def get_visualization_data(self):
        '''
        pyLDAvis prepare() function has multiple arguments. This function return all those variables for it. its a sperate function because it only return those needed variables and not store the vis.display output unlike visualize() function
        parameters:
            None : not needed. uses class variables        
        return:
            returns 5 variables which are needed to call prepare() function of pyVISlda. what these variables are i think clearly explained below and also in their original definition.
        '''

        topic_term_dists = self.topic_word_dist                  # shape (20, 46517)
        doc_topic_dists = self.document_topic_dist               # shape (2000, 20)
        doc_lengths = [len(i) for i in self.topic_assignments]   # shape (2000)  #number of words in each document
        vocab = self.vocab_list                                  # shape (46517)
        term_frequency = self.vocab_with_count                   # shape (46517)

        return topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency

    def visualize(self, output_html_file_path="output_visualization.html"):
        '''
        Using the pyVISlda, create an html output file to visualize the lda model.
        Parameters:
            output_html_file_path : string : path to the output target html file including the extension
        returns:
            None : saves the images 
        
        '''

        topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency = self.get_visualization_data()

        vis = pyLDAvis.prepare(topic_term_dists=topic_term_dists, doc_topic_dists=doc_topic_dists, doc_lengths=doc_lengths, vocab=vocab, term_frequency=term_frequency)
        vis_out = pyLDAvis.display(vis)
        with open(output_html_file_path, "w") as file:
            file.write(vis_out.data)
        print("HTML file saved at : ",output_html_file_path)
        
    def save_pickle(self, path, obj):
        f = open(path,"wb")
        pickle.dump( obj,f)
        f.close()

    def load_pickle(self, path):
        f = open(path,"rb")
        obj = pickle.load(f)
        return obj
    
    def save_variables(self, directory="", current_iteration=-1):
        '''
        function to save the variables used in gibbs_sampling function. function can be called when you want to save the variables for later use
        Parameters:
            directory : string: it is path to the folder in which the variables are to be stored. should include the last slash which is supposed to be a file name
        Returns:
            None
           Update: this function is still here and I am still saving all the variables separately for in case of we need only variables. and not the whole LDA object. I know its not required  for this assignment but I wrote it for generalizability 
        '''

        if directory=="":
            now = str(datetime.datetime.now())   # format : '2023-01-30 02:09:44.605562'
            #correcting the format so the name of the directory is correct
            now =  now.replace("-","_")
            now =  now.replace(":","_")
            now =  now.replace(" ","__")    # '2023_01_30__02_09_44.605562'
            
            if current_iteration == -1:
                class_info = str(self.num_iterations) +  "_" + str(self.num_topics) +  "_" + str(self.alpha) +  "_" + str(self.beta) 
                current_directory = str(Path.cwd())  + "/output/"
            else:
                class_info = str(self.num_iterations) +  "_" + str(self.num_topics) +  "_" + str(self.alpha) +  "_" + str(self.beta) +  "_" + str(current_iteration)
                current_directory = str(Path.cwd())  + "/intermediate_output/"

            directory = current_directory + now + "_" + class_info
            Path(directory).mkdir(parents=True, exist_ok=True)            # example folder name : 2023_01_31__05_30_01.002209_500_20_0.02_0.1_100
 
        lda_filename = "variable_lda.pckl"

        topic_assignments_filename = "variable_topic_assignments.pckl"
        topic_word_counts_filename = "variable_topic_word_counts.pckl" 
        document_topic_count_filename = "variable_document_topic_counts.pckl"
        topic_counts_filename = "variable_topic_counts.pckl" 

        document_topic_dist_filename = "variable_document_topic_dist.pckl" 
        topic_word_dist_filename = "variable_topic_word_dist.pckl" 
        vocab_list_filename = "variable_vocab_list.pckl" 

        lda_path = directory + "/" + lda_filename

        topic_assignments_path = directory + "/" + topic_assignments_filename
        topic_word_counts_path = directory + "/" + topic_word_counts_filename
        document_topic_count_path = directory + "/" + document_topic_count_filename
        topic_counts_path = directory + "/" + topic_counts_filename

        document_topic_dist_path = directory + "/" + document_topic_dist_filename
        topic_word_dist_path = directory + "/" + topic_word_dist_filename
        vocab_list_path = directory + "/" + vocab_list_filename
        
        self.save_pickle(lda_path, self)
        self.save_pickle(topic_assignments_path, self.topic_assignments)
        self.save_pickle(topic_word_counts_path, self.topic_word_counts)
        self.save_pickle(document_topic_count_path, self.document_topic_counts)
        self.save_pickle(topic_counts_path, self.topic_counts)
        self.save_pickle(document_topic_dist_path, self.document_topic_dist)
        self.save_pickle(topic_word_dist_path, self.topic_word_dist)
        self.save_pickle(vocab_list_path, self.vocab_list)


        print("Variables saved in the directory : ", directory )

    def load_variables(self, directory):
        '''
        function to load the variables used in gibbs_sampling function. function can be called when you want to continue training
        Parameters:
            directory : string: it is path to the folder in which the variables were stored. should NOT include the last slash which is supposed to be a file name
        Returns:
            None

            Update: this function is still here and I am still loading all the variables separately for in case of we need only variables. and not the whole LDA object. I know its not required  for this assignment but I wrote it for generalizability 
        '''
        topic_assignments_filename = "variable_topic_assignments.pckl"
        topic_word_counts_filename = "variable_topic_word_counts.pckl" 
        document_topic_count_filename = "variable_document_topic_counts.pckl"
        topic_counts_filename = "variable_topic_counts.pckl" 

        topic_assignments_path = directory + "/" + topic_assignments_filename
        topic_word_counts_path = directory + "/" + topic_word_counts_filename
        document_topic_count_path = directory + "/" + document_topic_count_filename
        topic_counts_path = directory + "/" + topic_counts_filename

        self.topic_assignments = self.load_pickle(topic_assignments_path)
        self.topic_word_counts = self.load_pickle(topic_word_counts_path)
        self.document_topic_counts = self.load_pickle(document_topic_count_path)
        self.topic_counts = self.load_pickle(topic_counts_path)

        print("Variables loaded from the directory : ", directory )
