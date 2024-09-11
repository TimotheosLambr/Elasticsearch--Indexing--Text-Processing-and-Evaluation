"""*****************************************************************************

                         aqu_answer_question_v2.py

This version has extra functions to make up a complete question answering system.

To run the complete system, do:
aqu_answer_the_questions()

This will answer one question in 'one_query_to_show_qa.json' (see aqu_read())

To classify queries:
aqu_keyword_classify_query()
aqu_classy_classify_query()

To use classy classifier, you must replace 'one_query_to_show_qa.json' with
'gold_standard_v2.json' (see aqu_read(). With just one query in .json, classy
will crash.

To find NEs and keywords in a text string:
aqu_find_nes_and_keywords()

To score NEs based on distance of keywords from query:
score_nes()

To use the classy classifier you need to do these in CMD:

pip install classy-classification
python -m spacy download en_core_web_md

The classy classifier also uses your .json which is assumed to be called
gold_standard_v2.json (see around line 44 below).

one_query_to_show_qa.json

one_query_to_show_qa.json contains only one query and demonstrates the basic QA
pipeline.

*****************************************************************************

"""

from elasticsearch import Elasticsearch
import json
import spacy
import classy_classification

# NB Classy needs model en_core_web_md not en_core_web_sm. If you have not got
# _md you can do in CMD:

# python -m spacy download en_core_web_md

es = Elasticsearch( "http://localhost:9200" )
# Create the client. Make sure Elasticsearch is already running!

"""-----------------------------------------------------------------------------

Read a gold standard JSON file. json.loads() reads a string containing a JSON
and converts it to a Python dictionary.

"""

def aqu_read():
    with open( '2001018.json', 'r', encoding = 'utf8' ) as f:
#    with open( 'gold_standard_v2.json', 'r', encoding = 'utf8' ) as f:
        s = f.read()
        d = json.loads( s )
    return d

"""-----------------------------------------------------------------------------

Shows how to process a dictionary containing a JSON and print out some information from it.

"""

def aqu_print():

    d = aqu_read()

    print( 'Number of queries: ', len( d[ "queries" ] ) )

    print( 'These are the queries:' )

    query_list = d[ "queries" ]
    for q in query_list:
        print( "Number:", q[ "number" ] )
        print( "Original query:", q[ "original_query" ] )
        # print( "Keyword query:", q[ "keyword_query" ] )
        # print( "Kibana query (might be empty):", q[ "kibana_query" ] )
        print( "answer_type:", q[ "answer_type" ] )
        # print( "Matching docs:", q[ "matches" ] )


"""-----------------------------------------------------------------------------

Gets data into this form and returns:

data = {
    "furniture": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "kitchen": ["There also exist things like fridges.",
                "I hope to be getting a new stove today.",
                "Do you also have some ovens."]
}

"""

def aqu_extract_data():

    d = aqu_read()
    query_list = d[ "queries" ]

    print( 'Number of queries: ', len( d[ "queries" ] ) )

    # Find the types used
    query_types = set()
    for q in query_list:
        query_types.add( q[ "answer_type" ] )

    print( 'Query types are:', query_types )
    # Checking for accidental duplicates

    # Create dictionary of training data with empty entries
    data = {}
    for type in query_types:
        data[ type ] = []

        # Populate the entries
    for q in query_list:
        data[ q[ "answer_type" ] ].append( q[ "keyword_query" ] )

    print( 'data is', data )

    return data


"""-----------------------------------------------------------------------------

Read .json and get a set of query types. Return set.

"""

def aqu_get_query_types():

    d = aqu_read()
    query_list = d[ "queries" ]

    # Find the types used
    query_types = set()
    for q in query_list:
        query_types.add( q[ "answer_type" ] )

    print( 'Query types are:', query_types )
    # Checking for accidental duplicates

    return query_types

"""-----------------------------------------------------------------------------

Classify queries using basic keyword matching "Who" -> person etc.

Uses SpaCy to tokenise query and return lemmas (root forms). Also brings down
to lower case.

NB: Make sure the types you classify into correspond to the ones used in the
.json!

We are getting the lemma for each word, i.e. in root form. e.g. 'sentences'
will become 'sentence'.

This classifier is super-basic but you will be able to reverse-engineer it on
your queries if all other classifiers fail.

"""

def aqu_keyword_classify_query( query ):

    nlp = spacy.load( "en_core_web_sm" )
    doc = nlp( query )
    tokens = [ token.lemma_ for token in doc ]
    # print( tokens )

    if 'who' in tokens:
        return( 'person' )
    elif 'where' in tokens:
        return( 'place' )
    elif 'when' in tokens:
        return( 'date' )
    else:
        return( 'reason' )

"""-----------------------------------------------------------------------------

Uses classy_classification module to classify query. This is a few-shot
classifier.

Before use you may need to do these in CMD:
pip install classy-classification
python -m spacy download en_core_web_md

Also, need to import spacy and classy_classification at the top of this file.

"""

def aqu_classy_classify_query( query ):

    data = aqu_extract_data()
    # Extract training data in correct Classy format from .json

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe(
        "classy_classification",
        config={
            "data": data,
            "model": "spacy"
        }
    )
    ans = nlp( query )._.cats
    # ans is a dict something like { 'animal': 0.13, 'date': 0.41, ... }
    print( ans )

    # Return the key in ans whose value is the largest, e.g. 'date'.
    return( max( ans, key=ans.get ) )

"""-----------------------------------------------------------------------------

Go through the queries in your Gold Standard, classify each with
aqu_keyword_classify_query(), compare the result with the Gold Standard, and
write out results exactly like this:

Query: Who is the keyboardist in the band Dire Straits
Gold type:  person 
Returned type: person
CORRECT

Query: ...
...

If the returned type is different, then it will be INCORRECT.
At the end, include a line like this with the accuracy:

Accuracy: 50%

"""

def aqu_extract_keywords_from_original_query(original_query):
    # Assuming necessary libraries are already imported and setup
    # This function should be the one you've implemented previously to extract keywords
    # For demonstration, we'll simplify it to just split the query into words
    return original_query.lower().split()

def aqu_keyword_classify_query(keywords):
    # Simple classification logic based on keywords
    # This should be replaced with your actual classification logic
    if 'who' in keywords or 'band' in keywords:
        return 'person'
    elif 'where' in keywords or 'capital' in keywords:
        return 'location'
    else:
        return 'unknown'

def aqu_evaluate_keyword_classify_query():
    # Gold Standard data
    gold_standard = [
        {"query": "Who is the keyboardist in the band Dire Straits", "type": "person"},
        {"query": "What is the capital of France", "type": "location"},
    ]
    
    correct_count = 0

    for entry in gold_standard:
        query = entry["query"]
        gold_type = entry["type"]
        
        # Extract keywords from the query
        keywords = aqu_extract_keywords_from_original_query(query)
        
        # Classify the query based on the extracted keywords
        returned_type = aqu_keyword_classify_query(keywords)
        
        # Determine correctness
        correctness = "CORRECT" if gold_type == returned_type else "INCORRECT"
        if correctness == "CORRECT":
            correct_count += 1
        
        print(f"Query: {query}")
        print(f"Gold type: {gold_type}")
        print(f"Returned type: {returned_type}")
        print(correctness)
        print()

    # Calculate and print the overall accuracy
    accuracy = (correct_count / len(gold_standard)) * 100
    print(f"Accuracy: {accuracy}%")

# Example call to the function
aqu_evaluate_keyword_classify_query()

"""-----------------------------------------------------------------------------

The input to this function is an original_query in text string form:
'Who played keyboards in the band Dire Straits'

The output is a list of keywords, minus stop words etc. 

play keyboard band dire straits

These keywords are going to be used for Elastic search and for scoring NEs
which are candidate answers.

"""

def aqu_extract_keywords_from_original_query( original_query ):

    
    nlp = spacy.load( "en_core_web_sm" )
    doc = nlp( original_query )
    tokens = [ token.lemma_ for token in doc ]

    # You need to study your queries first and then eliminate stopwords etc.
    # You may use a simple list of words to do this.

    tokens_minus_stopwords = tokens
    # This is the part you need to do

    return( tokens_minus_stopwords )

    # For now, return a fixed list:
    

    return( [  ] )
"""-----------------------------------------------------------------------------

Maps .json types like person, place, date, to SpaCy types like PERSON, GPE, DATE
"""

def aqu_convert_json_type_to_spacy_type( json_type ):

    conv = { 'person': 'PERSON',\
            'place': 'GPE',\
            'date': 'DATE' }

    if json_type in conv:
        return( conv[ json_type ] )
    else:
        return( 'PERSON' ) # Choose most frequent type

"""-----------------------------------------------------------------------------

paras is list of strings arising from Elastic search. number_of_paras is the
number we want. Returns that number of paras from paras, or all the paras in
paras if that is less.

"""

def aqu_select_paras( paras, number_of_paras ):

    if len( paras ) >= number_of_paras:

        return( paras[ 0 : number_of_paras ] )

    else:

        return( paras )

"""-----------------------------------------------------------------------------

keyword_result is coming back from the Elastic search (see
aqu_answer_the_questions()). query_search_requirements is also created there. It is a list of dicts like this:

[ { 'docs_required': 2, 'paragraphs_per_doc_required': 3 }, ... ]

In the call of this function, we have selected the dict within this which
corresponds to the query were are currently processing, parameter
search_requirement.

"""

def aqu_select_result_texts( keyword_result, search_requirement ):

    result_list = []
    # print( 'In aqu_select_result_texts():' )
    if len( keyword_result[ 'hits' ] [ 'hits' ] ) >= search_requirement[ 'docs_required' ]:
        for x in range( 0, search_requirement[ 'docs_required' ] ):

            paras = keyword_result[ 'hits' ] [ 'hits' ] [ x ] [ '_source' ] [ 'parsedParagraphs' ]
            result_list += aqu_select_paras( paras, search_requirement[ 'paragraphs_per_doc_required' ] )

            # print( keyword_result[ 'hits' ] [ 'hits' ] [ x ] )

    else:
        print( 'In aqu_select_result_texts():' )
        print( 'Asked for to many docs. Returning what there are.' )
        for x in range( 0, len( keyword_result[ 'hits' ] [ 'hits' ] ) ):

            paras = keyword_result[ 'hits' ] [ 'hits' ] [ x ] [ '_source' ] [ 'parsedParagraphs' ]
            result_list += aqu_select_paras( paras, search_requirement[ 'paragraphs_per_doc_required' ] )

            # print( keyword_result[ 'hits' ] [ 'hits' ] [ x ] )

    # print()
    return( result_list )

"""-----------------------------------------------------------------------------

Search for instances of entity_type (e.g. 'ORG') in text (which is a
sentence). Replace them by tuples which are numbered 1, 2, etc.

Also searches for keywords in the last passed, and replaces them by numbered
tuples as well.

This allows for multiple NEs of a given type in a sentence.

Example 1:

>>> aqu_find_nes_and_keywords( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'PERSON', [ 'company', 'ceo' ] )
['Mitsubishi', 'Electric', 'be', 'a', ('KEYWORD', 1, 'company'), 'and', ('PERSON', 1, ['John', 'Smith']), 'be', 'a', ('KEYWORD', 2, 'ceo'), '.']

Example 2:

>>> aqu_find_nes_and_keywords( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'ORG', [ 'company', 'ceo' ] )
[('ORG', 1, ['Mitsubishi', 'Electric']), 'be', 'a', ('KEYWORD', 1, 'company'), 'and', 'John', 'Smith', 'be', 'a', ('KEYWORD', 2, 'ceo'), '.']

"""

def aqu_find_nes_and_keywords( text, entity_type, keyword_lemma_list ):

    entity_serial_number = 0
    keyword_serial_number = 0
    posn = 0 # Word posn
    final_result = []

    def aqu_get_entity( doc ):

        nonlocal entity_serial_number
        nonlocal posn
        entity_word_list = []
        while posn < len( doc ) and doc[ posn ].ent_type_ == entity_type:

            entity_word_list.append( doc[ posn ].text )
            posn += 1
            # posn is now after the entity

        entity_serial_number += 1
        return( [ ( entity_type, entity_serial_number, entity_word_list ) ] )

    def aqu_get_non_entity( doc ):

        nonlocal keyword_serial_number
        nonlocal posn # Same posn as in outer function
        lemma_word_list = []
        while posn < len( doc ) and doc[ posn ].ent_type_ != entity_type:

            if doc[ posn ].lemma_ in keyword_lemma_list:
                keyword_serial_number += 1
                lemma_word_list += [ ( 'KEYWORD', keyword_serial_number, doc[ posn ].lemma_ ) ]
            else:
                lemma_word_list.append( doc[ posn ].lemma_ )

            posn += 1
            # posn is now after the word

        return( lemma_word_list )

    nlp = spacy.load( "en_core_web_sm" ) # Or _md?
    doc = nlp( text )    

    if doc[ posn ].ent_type_ == entity_type: # Start with entity of required type

        final_result += aqu_get_entity( doc )

    while posn < len( doc ):

        if posn < len( doc ):
            final_result += aqu_get_non_entity( doc )

        if posn < len( doc ):
            final_result += aqu_get_entity( doc )

    # print( final_result ) # NEs numbered and identified.
                          # Keywords numbered and identified.
    return( final_result )

"""-----------------------------------------------------------------------------

Scores NEs based on how close to keywords they are.

Input is a list like the one produced as output here:

Example 1:

>>> aqu_find_nes_and_keywords( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'PERSON', [ 'company', 'ceo' ] )
['Mitsubishi', 'Electric', 'be', 'a', ('KEYWORD', 1, 'company'), 'and', ('PERSON', 1, ['John', 'Smith']), 'be', 'a', ('KEYWORD', 2, 'ceo'), '.']

Example 2:

>>> aqu_find_nes_and_keywords( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'ORG', [ 'company', 'ceo' ] )
[('ORG', 1, ['Mitsubishi', 'Electric']), 'be', 'a', ('KEYWORD', 1, 'company'), 'and', 'John', 'Smith', 'be', 'a', ('KEYWORD', 2, 'ceo'), '.']

"""

def aqu_score_nes( sentence_tuple_list ):

    # [ x for x in sentence_tuple_list if type(x) is tuple and x[0] !='KEYWORD' ]

    posn = 0
    ne_list = []
    kw_list = []

    # Step 1: Find the NEs (i.e. they are not KEYWORD but instead PERSON, ORG etc
    # We find the position of each in the input list (0 is the first position).
    while posn < len( sentence_tuple_list ):

        if type( sentence_tuple_list[ posn ] ) is tuple and sentence_tuple_list[ posn ] [ 0 ] != 'KEYWORD':

            ne_list.append( ( sentence_tuple_list[ posn ], posn ) )

        posn += 1

    print( 'Named Entities listed with position of each (1st word is number 0):' )
    # Example:
    # ('PERSON', 1, ['Alan', 'Clark']), 0)
    # Syntax: ( <named_entity_type>, <ordinal_number>, <words_in_NE>, 
    #           <position_in_the_list> )

    # PERSON means the NE type (relative to SpaCy).
    # Number 1 means this NE instance is ordinal number 1 (i.e. it is the first 
    # instance of an NE in the sentence).
    # If another PERSON comes up in the sentence, it will be number 2 and so on.
    # Even repeat instances of the very same PERSON will have different numbers.
    # ['Alan', 'Clark'] is the words in the NE.
    # Finally, 0 means the position of the NE in the list is zero, i.e. it is the
    # first element of the list.

    print( ne_list )
    print()

    posn = 0 # Reset to go through again
    # Step 2: Find the keywords (i.e. they are KEYWORD and not PERSON, ORG etc
    # We find the position of each in the input list (0 is the first position).
    while posn < len( sentence_tuple_list ):

        if type( sentence_tuple_list[ posn ] ) is tuple and sentence_tuple_list[ posn ] [ 0 ] == 'KEYWORD':

            kw_list.append( ( sentence_tuple_list[ posn ], posn ) )

        posn += 1

    print( 'KEYWORDS listed with position of each:' )
    print( kw_list )
    print()

    # NOTE: Output of ne_list could be:
    # [(('ORG', 1, ['Mitsubishi', 'Electric']), 0), (('ORG', 2, ['Microsoft', 'Corporation']), 11)]


    # Note: Output of kw_list could be:
    # [(('KEYWORD', 1, 'company'), 3), (('KEYWORD', 2, 'ceo'), 9)]

    # Step 3: Take each NE in turn and score it against all the keywords:

    score = 0

    print( 'NE(s) in the sentence with contribution of each KEYWORD to NE score:' )

    for ne in ne_list:

        ne_posn = ne[ 1 ]
        print( 'NE is', ne, 'posn is', ne_posn )
        score = 0
        for kw in kw_list:

            contribution = abs( ne_posn - kw[ 1 ] )
            print( 'Contribution of', kw, 'is', contribution )
            score += contribution

        print( 'So, score for', ne, 'is', score )
        print()

    print( '(Lower NE score means better match but zero score means no KEYWORDs in sentence.)' )


"""-----------------------------------------------------------------------------

text is a sentence (string)
entity_type is the NE sought (e.g. PERSON, ORG etc, according to SpaCy NE tagger)

keyword_lemma_list is a list of other keywords from the query we want to match
as well. These should be in lemma form, all lower case.

What is does:

1. Finds the NEs (if any) and the posn of each.

2. Finds the keywords (if any) and the posn of each

3. Scores each NE based on its distance from the keywords.

Note: The score is a simple sum of the distances of each NE from each
keyword. The lower the score, the better the match, with 1 being the minimum
match. A score of 0 means there are no keywords found in the sentence.

Example call:

aqu_find_ans( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'PERSON', [ 'company', 'ceo' ] )

aqu_find_ans( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'ORG', [ 'company', 'ceo' ] )

"""

def aqu_find_ans( text, entity_type, keyword_lemma_list ):

    ans = aqu_find_nes_and_keywords( text, entity_type, keyword_lemma_list )

    aqu_score_nes( ans )

"""-----------------------------------------------------------------------------

A test function to find out the exact effect of 'nonlocal' on inner and outer
functions. Specifically, a variable in the outer function which you want to
change in the inner function.

"""

def aqu_test():

    n = 1
    def inside():
        nonlocal n    
        n += 1

    print( n )
    inside()
    print( n )

# Here are some print statements which demonstrate the two classifiers and
# a method of answer selection.

"""-----------------------------------------------------------------------------

Goes through the questions in your Gold Standard. Carries out these steps:

1. Start with a query (NB use original_query form of your queries):
Who played keyboards in the band Dire Straits

2. Extract keywords from original_query for elastic search:
play keyboard band dire straits

NB: You need to eliminate function words like prepositions ( 'in' ) and
determiners 'the' to avoid false matches to documents.

3. Find the expected answer type:
person (in our .json)
PERSON (in SpaCy)

4. Search with elastic to find matching documents

5. From those documents find supporting sentences

6. Search for NEs in the supporting sentence using SpaCy

7. Look for sentences which contain at least one NE of correct type (i.e. PERSON in our example)

8. In sentences, identify NEs and also identify relevant keywords (see keywords we extracted above)

9. Score all NEs based on their distance from relevant keywords

10. Select the NE with the best score

11. Return it as the answer!

"""

def aqu_answer_the_questions():

    d = aqu_read()
    query_list = d[ "queries" ]
    print()
    print( 'Number of queries in Gold Standard:', len( query_list ) )

    # There is one line below for each of your 20 queries. You can set the
    # number of documents to be searched for answers. Within those docs, you
    # can specify the number of paras to look at. At present, both values are
    # set to 1 for each of the 20 queries. Thus, it will look at just the first
    # para in the first doc, for each query, exactly as it did before.
    # However, you can experiment with other values, based on your investigations
    # with  Kibana searches to find out where the answers actually are.

    query_search_requirements = [
        { 'docs_required': 2, 'paragraphs_per_doc_required': 3 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 },
        { 'docs_required': 1, 'paragraphs_per_doc_required': 1 }
        ]        

    for q, r in zip( query_list, query_search_requirements ):
        print()
        print( '--------------------------------------------------------------------------' )
        print()

        # STEP 1
        original_query = q[ "original_query" ]
        keyword_query = q[ "keyword_query" ]
        print( 'original_query:' )
        print( original_query )
        print()

        # STEP 2
        search_tokens = aqu_extract_keywords_from_original_query( original_query )       # Use this form for answer NE scoring (see below)

        search_tokens_string = ' '.join( search_tokens )
        # Use this form for Elastic search (also see below)

        print( 'search_tokens extracted:' )
        print( search_tokens )
        print()

        # STEP 3
        query_type_json = aqu_keyword_classify_query( original_query )
        query_type_spacy = aqu_convert_json_type_to_spacy_type( query_type_json )
                           # Not properly written yet!
        print( 'query_type_json:', query_type_json )
        print( 'query_type_spacy:', query_type_spacy )
        print()

        # STEP 4
        if search_tokens_string != '':

            query = { "multi_match": { "query": search_tokens_string, "fields": [], "type": "best_fields"}}
            print( 'Elastic query submitted with search_tokens_string:' )
            print( query )
            print()
            keyword_result = es.search(
                index = 'student_index_v3',
                size = 10, # Max number of hits to return. Default is 10.
                query = query)
        else:
            print( 'Could not submit search_tokens_string=''' )
            keyword_result = []

        # STEP 5
        # result_text = keyword_result[ 'hits' ] [ 'hits' ] [ 0 ] [ '_source' ] [ 'parsedParagraphs' ] [ 0 ]
        # Note: This is just selecting the first sentence from the first 
        # matching document
        # print( 'First sentence from first document returned by Elastic:' )
        # print( result_text )
        # print()

        result_text_list = aqu_select_result_texts( keyword_result, r )
        print( 'result_text_list:' )
        print( result_text_list )
        print()

        # STEP 6, 7, 8 - Commented out. This was the old way. One para only.
        # nes_and_keywords = aqu_find_nes_and_keywords( result_text, query_type_spacy, search_tokens )

        # print( 'Named Entities recognised and search_tokens marked as KEYWORD:' )
        # print( nes_and_keywords )
        # print()

        # STEP 9
        # aqu_score_nes( nes_and_keywords )

        # STEP 6, 7, 8 now done multiple times, for each para in result_text_list
        for result_text in result_text_list:

            nes_and_keywords = aqu_find_nes_and_keywords( result_text, query_type_spacy, search_tokens )

            print()
            print( '=========================================' )
            print( 'Looking at a para, searching for answers:' )
            print( '=========================================' )
            
            print( 'Named Entities recognised and search_tokens marked as KEYWORD:' )
            print( nes_and_keywords )
            print()

            aqu_score_nes( nes_and_keywords )



        # STEP 10 is implicit. Just prints out the scores. You choose the largest.
aqu_answer_the_questions()

#print()
#print( 'Input to keyword classifier is', '\'Who is the CEO?\'' )
#print( aqu_keyword_classify_query( 'Who is the CEO?' ) )
#print()

#print()
#print( 'Input to keyword classifier is', '\'Where is Mitsubishi Electric?\'' )
#print( aqu_keyword_classify_query( 'Where is Mitsubishi Electric?' ) )
#print()

#print()
#print( 'Input to classy classifier is', '\'Who is the CEO?\'' )
#print( aqu_classy_classify_query( 'Who is the CEO?' ) )
#print()

#print()
#print( 'Input to classy classifier is', '\'Where is Mitsubishi Electric?\'' )
#print( aqu_classy_classify_query( 'Where is Mitsubishi Electric?' ) )
#print()

#print()
#print( 'Inputs to aqu_find_ans() are' )
#print( 'Mitsubishi Electric is a company and John Smith is a CEO.' )
#print( 'PERSON' )
#print( [ 'company', 'ceo' ] )
#print()
#print( 'Results:' )
#aqu_find_ans( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'PERSON', [ 'company', 'ceo' ] )
#print()

#print()
#print( 'Inputs to aqu_find_ans() are' )
#print( 'Mitsubishi Electric is a company and John Smith is a CEO.' )
#print( 'ORG' )
#print( [ 'company', 'ceo' ] )
#print()
#print( 'Results:' )
#aqu_find_ans( 'Mitsubishi Electric is a company and John Smith is a CEO.', 'ORG', [ 'company', 'ceo' ] )
