TEMPLATES = {

########## MAIN TEMPLATE for EXPLANATION SELECTION
"template":"""
You are an AI assistant that analyzes struggling students behavior to help them in their learning trajectories and facilitate learning in the best possible way. 
You have many information to help you in your goal:
- A model prediction of student performance at the end of the course, in the form of “pass” or “fail”.
- A post-hoc explainable AI approach that identifies which features are important to this student’s prediction.
- Data in the form of student's features over 5 weeks that were used by the model. You will always see the most relevant features selected by the explainer
- The course content and structure.
- Detailed instructions on how to reason.

{model_description}
{features_description}  
{post_hoc_description}
{course_description}

Take into consideration this data:

{data_string}

INSTRUCTIONS:
{instructions}

QUESTION: {goal_definition}

""",

### Variables that will be filled in the template above

# Post hoc explainer description
"post_hoc_description" : {
    "LIME": "We use LIME as our explainable AI approach, which gives importance scores for 20 features that contributed the most to the prediction of the model. Positive scores positively contribute to the model's decision to reach the predicted outcome. Negative scores would push the prediction towards the opposite class. The magnitude of the score indicates the strength of the feature's contribution.",
    
    "MC_LIME": "We use Minimal Counterfactual LIME as our explainable AI approach, which finds the smallest number of changes necessary to change a prediction from student failure to student success. It uses LIME scores to select the features to change. The output are the sets of features with the new values that would change the prediction.",

    "CEM": "We use CEM Counterfactuals as our explainable AI approach, which finds the smallest number of changes necessary to change a prediction from student failure to student success. The output is the minimal difference in the feature values that would change the prediction."
},

# Goal Definition
"goal_definition" : "Given the information above, follow precisely the instructions above and write a small report on what you found. Only use the results from the explainable AI approach and the student's behavior data to justify your conclusions.",

# Model Description
"model_description" : "The model you are using is a recurrent neural network that is trained on predicting the student performance at the end of the course, in the form of “pass” or “fail”. The features of that the model are using are derived from student behavior:",

# Course Description
"course_description_dsp_001" : """
The course the student is taking is Digital Signal Processing 1, which is a Master’s level course over 10 weeks under the topic of Electrical Engineering and Computer Science.
This is the course content:

WEEK 1

SKILLS: Digital Signals
TOPICS: Welcome to the DSP course, Introduction to signal processing

WEEK 2

SKILLS: Digital Signals
TOPICS: Discrete time signals, The complex exponential, The Karplus-Strong Algorithm

WEEK 3

SKILLS: Hilbert (Linear Alg.)
TOPICS: Motivation and Examples, From Euclid to Hilbert, Hilbert Space, properties and bases, Hilbert Space and approximation

WEEK 4

SKILLS: DFT, DTFT DFS, DTFT: intuition and properties, FFT
TOPICS: Exploration via a change of basis, The Discrete Fourier Transform (DFT), DFT Examples, DFT, DFS, DTFT, DTFT formalism, Relationship between transforms, Sinusoidal modulation, FFT: history and algorithms

WEEK 5

SKILLS: Ideal Filters, Filter Design
TOPICS: Linear Filters, Filtering by example, Filter stability, Frequency response, Ideal filters, Filter design - Part 1: Approximation of ideal filters, Realizable filters, Implementation of digital filters, Filter design - Part 2: Intuitive filters, Filter design - Part 3: Design from specs, Real-time processing, Dereverberation and echo cancelation

WEEK 6

SKILLS: Modulation, Interpolation & Sampling
TOPICS: Introduction to continuous-time paradigm, Interpolation, The space of bandlimited signals, Sampling and aliasing: Introduction, Sampling and aliasing, Discrete-time processing and continuous-time signals, Another example of sampled acquisition

WEEK 7

SKILLS: Multirate
TOPICS: Stochastic signal processing, Quantization, A/D and D/A conversion

WEEK 8

SKILLS: DFT, DTFT DFS, Ideal Filters
TOPICS: (Revisiting the topics of week 4 with additional context) Image processing, Image manipulations, Frequency analysis, Image filtering, Image compression, The JPEG compression algorithm

WEEK 9

SKILLS: Modulation, Quantization
TOPICS: Digital communication systems, Controlling the bandwidth, Controlling the power, Modulation and demodulation, Receiver design, ADSL

WEEK 10

SKILLS: Applications
TOPICS: The end, Startups and DSP, Acknowledgements, Conclusion video
""",

"course_description_villesafricaines_001" : """
Course Title: Villes africaines I: Introduction à la planification urbaine
Overview: This course explores various aspects of urban development in Africa. Divided into weekly chapters, each week focuses on different topics and skills related to urban planning and development.

WEEK 1

TOPICS: Introduction, Urbanisation in Africa (Parts 1 & 2), What is urban planning? (Parts 1 & 2), African cities (Parts 1 & 2)
QUIZ: Quiz 1

WEEK 2

TOPICS: Stakes and challenges (Parts 1 & 2), Which model for which city (Parts 1 & 2), Shape of urbanization
QUIZ: Quiz 2

WEEK 3

TOPICS: Globalization and cities, Urban Sustainable Development, Climate Change, Urban Environment, Transport and Mobility
QUIZ: Quiz 3

WEEK 4

TOPICS: Urban facilities, Public spaces, Habitat, Land issue
QUIZ: Quiz 4

WEEK 5

TOPICS: Urban sprawl, Professions of the city, Land readjustment strategies, Informal settlements upgrading
QUIZ: Quiz 5

WEEK 6

TOPICS: Critical reading, The 10 basic principles, The urban form, Accessibility and mobility, Infrastructures, Urban agriculture, Climate change
QUIZ: Quiz 6

WEEK 7

TOPICS: Tools of urban planning, Diagnostic, Basic Data, GIS, Case study: Ndjamena
QUIZ: Quiz 7

WEEK 8

TOPICS: Introduction to the week: subdivisions, Basic principles of subdivisions, Making networks profitable, Subdivision "step by step", Land charge and real estate investments, Examples of subdivisions
QUIZ: Quiz 8

WEEK 9

TOPICS: Slums, Precarious neighborhoods: restructuring, Precarious neighborhoods: census, Precarious neighborhoods: housing program
QUIZ: Quiz 9

WEEK 10

TOPICS: Financing, Planning and financial resources, Addressing, Cadaster


WEEK 11

TOPICS: Reading the Images, Approaches of the Urban Governance, Decentralization, Communities


WEEK 12

TOPICS: Measuring impact, Synthesis and conclusions, End of the course
QUIZ: Final questionnaire
""",

"course_description_geomatique_003" : """
Course Title: Éléments de Géomatique
Overview: This course delves into the field of geomatics, focusing on geodesy, cartography, and geographic information systems (GIS). It is structured into weekly chapters, each concentrating on different aspects and techniques of geomatics.

WEEK 1

SKILLS: Introduction to Geomatics
TOPICS: Introduction to the course, Representation, Acquisition, Management
QUIZZES: Quiz: Introduction to Geomatics

WEEK 2

SKILLS: Geodesy
TOPICS: Introduction to Geodesy, Units, Coordinate Systems, Exercise on infinitesimal displacement
QUIZZES: Quiz: Geodetic Principles, Quiz: Earth Coordinates, Quiz: Geodetic Units

WEEK 3

SKILLS: Geodetic References
TOPICS: Geodetic References, Projections
QUIZZES: Quiz: Projections, Quiz: Swiss Coordinates

WEEK 4

SKILLS: Cartography
TOPICS: Introduction to Cartography, Semiology, Exercise on slab thickness
QUIZZES: Quiz: Cartography, Quiz: Semiology

WEEK 5

SKILLS: Modeling and Interpolation
TOPICS: Modeling and Interpolation
QUIZZES: Quiz: Modeling, Quiz: Introduction to DEM (Digital Elevation Model)

WEEK 6

SKILLS: Geometric Leveling
TOPICS: Introduction to Geometric Leveling, Definitions of Altitudes, Measurement Principles, Control Level, Progression, Reading on the Leveling Rod
QUIZZES: Quiz: Instruments and Measurements

WEEK 7

SKILLS: Surveying
TOPICS: Introduction to Surveying, Orientation, Step-by-step Calculation of Bearing, Step-by-step Calculation of Station Orientation
QUIZZES: Quiz: Orientations and Bearings, Quiz: Instruments and Measurements for Surveying

WEEK 8

SKILLS: Polar Surveying
TOPICS: Polar Surveying, Theodolite, Representation of Relief / Geomorphometry, Exercise on Pointing, Step-by-step Calculation of a Polar Survey
QUIZZES: Quiz: Triangulation

WEEK 9

SKILLS: Distance Measurement Techniques
TOPICS: Electronic Distance Measurement, Trigonometric Leveling
QUIZZES: Quiz: Measurement of Distances

WEEK 10

SKILLS: Satellite Positioning Principles
TOPICS: Principles of Localization by Satellites
QUIZZES: Quiz: GPS Principles, Quiz: GPS and DOP (Dilution of Precision)

""",

######################### PRESENTATION PROMPT TEMPLATES

"presentation_examples" : "",

"presentation_template" : """
Given this report, I want you to write a shorter version using the theory of feedback from Hattie et al.:

Where Am I Going? - A brief description of the student's performance and explicitly state the learning goal
How Am I Doing? - A brief description of the explanation findings
Where to Next? - Two recommended actions that the student can take that are specific to weeks of the course (make connections between previous weeks and upcoming weeks)

The student who is going to interact with you is the same student that the data is about, so use the tone of a professor who is talking to a student in a comforting and personable manner. 

Follow the instructions underneath in the INSTRUCTIONS section.

INSTRUCTIONS In the explanation findings section (How am I going?) explicitly use the following structure:
{presentation_instruction}

{course_description}

5 weeks of the course have concluded.

Follow these rules:

do not include the output of the model or a prediction
if you include a feature name, describe what it means
try to be as concise as possible
do not include the headers from hattie et al. in the response, but keep the structure
limit yourself to 200 words
do not include a sign-off, simply include the main content

To communicate this intervention most effectively, use Grice's maxims of conversation.

do not say things that you believe to be false
do not say things for which you do not have sufficient evidence.
do not add information that is not relevant
Only say what is relevant
be orderly
Your goal is to explain to the student what they should do to improve their performance in the course in the best way possible. Follow the instructions above.

""",

###### FORMAT TEMPLATES
"format_instructions": "Return a JSON file with a string with your feedback to the student.",

"conversation_template" : """
Current conversation:
{chat_history}
{input}
AI Assistant:
"""
}