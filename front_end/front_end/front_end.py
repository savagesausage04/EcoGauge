import reflex as rx
from webscrapyer import scrape_threads
import json
import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from collections import Counter
from typing import List
from reflex_pyplot import pyplot



import google.generativeai as genai
import os
from webscrapyer import thread_list

colors = {
    "background": "#E6E6FA",  # Lavender purple
    "text": "#333333",  # Dark gray for main text
    "accent_blue": "#007acc",  # Blue for header background and accents
    "accent_green": "#A8E6CE",  # Light green for some text and button
    "input_bg": "#f0f0f0",  # Light gray for input background
    "input_text": "#333333",  # Dark gray for input text
    "hover_teal": "#2BB673",  # Darker teal for button hover
    "white": "#FFFFFF",
    "light_green": "#6cc767",

}

button_hover_effects = """
    <style>
        .hover-button {
            background-color: #A8E6CE;
            transition: background-color 0.3s;
        }

        .hover-button:hover {
            background-color: #2BB673;
        }
    </style>
"""

custom_css = """
@keyframes scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
"""

fruits = [
    "Pollution",
    "☘︎",
    "Ecosystems",
    "☘︎",
    "Biodiversity",
    "☘︎",
    "Footprint",
    "☘︎",
    "Recycle",
    "☘︎",
    "Agriculture",
    "☘︎",
    "Resources",
    "☘︎",
    "Energy",
    "☘︎",
    "Solar",
    "☘︎",
    "Geothermal",
    "☘︎",
    "Emissions",
    "☘︎",
    "Global Warming",
    "☘︎",
    "Environmental Activism",
    "☘︎",
    "Urban Planning",
    "☘︎",
    "Climate",
    "☘︎",
    "Carbon Neutral",
    "☘︎",
    "Biomass",
    "☘︎",
    "Wind Power",
    "☘︎",
    "Electricity",
    "☘︎",
    "Greenhouse Gas",
    "☘︎",
]


class State(rx.State):
    business_id: str = ""
    show_results: bool = False
    next_steps: List[str] = []
    sentiment_summary: str = ""

    plot_figure_data = [5, 5, 5]
    plot_figure_labels = ["Negative", "Positive", "Neutral"]
    saved_figure = False
    show_about_us: bool = False

    def toggle_about_us(self):
        self.show_about_us = not self.show_about_us

    # def get_pyplot(self):
    #     data: dict = {}
    #     pyplot.

    @rx.var(cache=True)
    def pie_maker(self) -> plt.Figure:
        labels = self.plot_figure_labels
        sizes = self.plot_figure_data

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels)
        return fig
    

    def submit(self):
        if self.business_id:
            self.show_results = True

            topic = self.business_id
            topics = topic.split()

            if len(topics) == 1:
                url = f"https://www.threads.net/search?q={topic}&serp_type=default"
            elif len(topics) == 2:
                url = f"https://www.threads.net/search?q={topics[0]}%20{topics[1]}&serp_type=default"

            thread_list = scrape_threads(url)

            class Sentiment:
                NEGATIVE = "Negative"
                POSITIVE = "Positive"
                NEUTRAL = "Neutral"

            class Review:
                def __init__(self, text, score):
                    self.text = text
                    self.score = score
                    self.sentiment = self.get_sentiment()

                def get_sentiment(self):
                    if self.score <= 2:
                        return Sentiment.NEGATIVE
                    elif self.score == 3:
                        return Sentiment.NEUTRAL
                    else:
                        return Sentiment.POSITIVE

            class ReviewContainer:
                def __init__(self, reviews):
                    self.reviews = reviews

                def get_text(self):
                    return [x.text for x in self.reviews]

                def get_sentiment(self):
                    return [x.sentiment for x in self.reviews]

                def evenly_distribute(self):
                    negative = list(
                        filter(
                            lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews
                        )
                    )
                    positive = list(
                        filter(
                            lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews
                        )
                    )
                    neutral = list(
                        filter(lambda x: x.sentiment == Sentiment.NEUTRAL, self.reviews)
                    )

                    positive_shrunk = positive[: len(negative)]
                    self.reviews = negative + positive_shrunk + neutral
                    random.shuffle(self.reviews)

            file_name = "/Users/aaronnguyen/Desktop/Books_small_10000.json"

            yelp_reviews = thread_list

            reviews = []

            with open(file_name) as f:
                for line in f:
                    review = json.loads(line)

                    reviews.append(
                        Review(review["reviewText"], review["overall"])
                    )  # creates a list with Review objects with text and score attributes

            training, testing = train_test_split(
                reviews, test_size=0.9, random_state=42
            )
            train_container = ReviewContainer(training)
            test_container = ReviewContainer(testing)

            train_container.evenly_distribute()
            training_x = train_container.get_text()
            training_y = train_container.get_sentiment()

            test_container.evenly_distribute()
            testing_x = test_container.get_text()
            testing_y = test_container.get_sentiment()

            training_y.count(Sentiment.POSITIVE)
            training_y.count(Sentiment.NEGATIVE)

            # vectorizes training and testing text data into numerical format
            vectorizer = TfidfVectorizer()
            training_x_vectors = vectorizer.fit_transform(training_x)
            # testing_x_vectors = vectorizer.transform(testing_x)

            # scikit learn svm model
            clf_svm = svm.SVC(kernel="linear")
            clf_svm.fit(training_x_vectors, training_y)
            # clf_svm.predict(testing_x_vectors[0])

            # scikit decision tree model
            """
            clf_dec = DecisionTreeClassifier()
            clf_dec.fit(training_x_vectors, training_y)
            clf_dec.predict(testing_x_vectors[0])

            #scitkit gaussian naive bayes model
            clf_gnb = GaussianNB()
            training_x_vectors_dense = training_x_vectors.toarray()
            testing_x_vectors_dense = testing_x_vectors.toarray()
            clf_gnb.fit(training_x_vectors_dense, training_y)
            clf_gnb.predict(testing_x_vectors_dense[0].reshape(1, -1))

            #scikit logistic regression model
            clf_log = LogisticRegression()
            clf_log.fit(training_x_vectors, training_y)
            clf_log.predict(testing_x_vectors[0])
            """

            # prints the accuracy of each model
            """
            print(f'This is the initial svm model accuracy: {clf_svm.score(testing_x_vectors, testing_y)}')
            print(f'This is the initial dec model accuracy: {clf_dec.score(testing_x_vectors, testing_y)}')
            print(f'This is the initial gnb model accuracy: {clf_gnb.score(testing_x_vectors_dense, testing_y)}')
            print(f'This is the initial log model accuracy: {clf_log.score(testing_x_vectors, testing_y)}')
            """

            # f1_score(testing_y, clf_svm.predict(testing_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])

            # yelp_reviews = ['the customer service was not that great given the fact that they served me raw chicken', "wow this restaurant is really favorable", "horrible waste of time"]
            new_test = vectorizer.transform(yelp_reviews)
            print("\n")
            sentiment_list = clf_svm.predict(new_test)
            print(
                f"This is the prediction results for the sample test set: {sentiment_list}"
            )

            """
            #parameter tuning to increase model accuracy
            parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16, 32)}
            svc = svm.SVC()
            clf = GridSearchCV(svc, parameters, cv=5)
            clf.fit(training_x_vectors, training_y)
            print('\n')
            #print(f'This is the svm accuracy post parameter tuning: {clf.score(testing_x_vectors, testing_y)}')
            """
            sentiment_counts = Counter(sentiment_list)
            total_sentiments = len(sentiment_list)
            proportions = {
                label: count / total_sentiments
                for label, count in sentiment_counts.items()
            }
            proportions_list = list(proportions.values())
            keys = ["Negative", "Positive", "Neutral"]
            print(proportions_list)

            
    
            self.plot_figure_data = proportions_list
            self.saved_figure = True

            # plt.title("Distribution of Media Sentiment towards Sustainability Topic")
            # plt.savefig('assets/sentiment_chart.png')
            # plt.close()

            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Here are some of the comments related to {topics}. Take some of the criticisms and offer some points of future improvements. Make the feedback short with 3 improvements and each one sentence. The text that follows are the comments: {yelp_reviews}. Don't bold the titles please"
            )
            response_text = response.text



            negative = proportions_list[0]
            positive = proportions_list[1]
            neutral = proportions_list[2]

            if negative >= positive:
              self.sentiment_summary = f'The public sentiment for {topic} is overwhelmingly negative with {negative:.1%} negative.'
            elif positive > negative:
                self.sentiment_summary = f'The public sentiment for {topic} is overwhelmingly positive with {positive:.1%} positive.'
            else:
                self.sentiment_summary = f'The public sentiment for {topic} is overall neutral with {neutral:.1%} neutral.'

            
            



            self.next_steps = [
                step.strip() for step in response_text.split("\n") if step.strip()
            ]

    def go_back(self):
        self.show_results = False
        self.business_id = ""

    def handle_key_press(self, key: str):
        if key == "Enter":
            self.submit()

    def get_search_text(self):
        pass

    

    


def header():
    return rx.box(
        rx.hstack(
            rx.heading(
                "EcoGauge", color=colors["accent_green"], font_size="1.5em"
            ),
            rx.spacer(),
            rx.button(
                "About Us",
                on_click=State.toggle_about_us,
                background=colors["accent_green"],
                color=colors["text"],
                padding="0.5em 1em",
                border_radius="0.3em",
                class_name="hover-button",
            ),
            width="100%",
            padding="0 1em",
        ),
        width="100%",
        padding="0.5em",
        background=colors["accent_blue"],
    )

def about_us_page():
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "About Us",
                    color=colors["accent_blue"],
                    font_size="2.5em",
                    margin_bottom="0.2em",
                ),
                rx.spacer(),
                rx.button(
                    "Back to Home",
                    on_click=State.toggle_about_us,
                    background=colors["accent_green"],
                    color=colors["text"],
                    padding="0.5em 1em",
                    border_radius="0.3em",
                    class_name="hover-button",
                ),
                width="100%",
                padding="1em",
            ),
            rx.box(
                rx.text(
                  """1. Who this helps:

a) Environmental Activists & Organizations: EcoGauge empowers sustainability advocates, nonprofits, and researchers by providing real-time insights into public sentiment on environmental issues. It helps them stay informed about public opinion, identify areas for improvement in communication, and tailor their strategies based on public perception.

b) Policymakers & Corporations: It enables policymakers and businesses with sustainable initiatives to understand public concerns or enthusiasm regarding specific environmental topics (e.g., renewable energy, climate change, sustainable farming). By identifying positive and negative trends, these stakeholders can align their policies or business strategies with public interests.

c) General Public: The tool offers transparency, allowing the public to see how environmental topics are perceived on a larger scale and encouraging more informed discussions about sustainability issues.

2. Why is this impactful:

a) Driving Informed Change: EcoGauge helps amplify voices in the sustainability space by surfacing sentiment data that can influence decisions and communication strategies. For example, if negative sentiment around sustainable farming is detected, advocates can intervene with better education or outreach efforts to clarify misconceptions.

b) Enabling Efficient Action: With clear visual insights like sentiment pie charts, organizations can prioritize issues based on real-time feedback and act on the public's concerns more efficiently, driving faster solutions to environmental challenges.

c) Facilitating Feedback Loops: The Gemini feedback mechanism adds value by directly engaging with the sentiment data, generating automatic feedback or responses to common concerns. This creates a continuous loop where organizations and the public can interact with real-time data.

3. Why is this unique:

a) Sentiment + Feedback in One: Unlike many sentiment analysis tools that only provide data, EcoGauge takes it further by using Gemini's feedback to address the underlying concerns in threads. This dual approach enables proactive engagement rather than passive observation.

b) Niche Focus on Sustainability: By specializing in environmental and sustainable topics, EcoGauge targets a vital global issue and becomes an indispensable tool for those working toward positive environmental impact.

c) Web Scraping Precision: The combination of thread scraping and sentiment analysis gives a more grassroots view of public opinion, capturing the nuances of how everyday people are talking about sustainability across forums and social platforms.""",
                    color=colors["text"],
                    white_space="pre-wrap",
                ),
                background=colors["input_bg"],
                padding="2em",
                border_radius="0.5em",
                width="90%",
                max_width="800px",
                height="70vh",
                overflow="auto",
                border=f"1px solid {colors['accent_blue']}",
            ),
            width="100%",
            max_width="1000px",
            align_items="center",
            justify_content="center",
            padding="2em",
            spacing="1em",
        ),
        height="100vh",
        width="100%",
        background=colors["background"],
    )

def scrolling_text():
    return rx.box(
        rx.hstack(
            rx.text("                ".join(fruits + fruits), white_space="nowrap", color="black"),
            rx.text("                ".join(fruits + fruits), white_space="nowrap", color="black"),
            animation="scroll 120s linear infinite",
            width="fit-content",
        ),
        overflow="hidden",
        width="100%",
        background="rgba(0, 122, 204, 0.1)",
        padding="10px 0",
        position="absolute",
        bottom="0",
        left="0",
    )




def home_page():
    return rx.center(
        rx.vstack(
            rx.text(
                "Curious about any environmental topics? ",
                color=colors["text"],
                font_size="1.5em",
                margin_bottom="1em",
            ),
            rx.input(
                placeholder="Environmental Topic: ",
                on_change=State.set_business_id,
                on_key_down=State.handle_key_press,
                value=State.business_id,
                width="100%",
                padding="0.5em",
                border_radius="0.3em",
                border=f"1px solid {colors['accent_blue']}",
                background=colors["input_bg"],
                color=colors["input_text"],
                _placeholder={"color": colors["text"], "opacity": 0.6},
            ),
            rx.button(
                "Review",
                on_click=State.submit,
                background=colors["accent_green"],
                color=colors["text"],
                padding="0.5em 2em",
                border_radius="0.3em",
                margin_top="1em",
                class_name="hover-button",
            ),
            scrolling_text(),
            spacing="1em",
            width="80%",
            max_width="600px",
            align_items="center",
            padding="2em",
            background="rgba(255, 255, 255, 0.8)",
            border_radius="1em",
            margin_top="40px",
            height="300px",
            position="relative",
        ),
        height="100vh",
        width="100%",
        background_image="url('/background.jpeg')",
        background_size="cover",
        background_position="center",
        background_repeat="no-repeat",
        background_attachment="fixed",
    )




# def make_pyplot_graph():
#     fig, ax = plt.subplots()
#     # plt.pie(State.plot_figure_data, labels=State.plot_figure_labels, autopct="%1.1f%%",startangle=90)
#     plt.pie([34, 33, 33], labels=State.plot_figure_labels, autopct="%1.1f%%",startangle=90)
#     plt.title("Distribution of Media Sentiment towards Sustainability Topic")
#     plt.close()
#     return pyplot(fig)



def results_page():
    return rx.center(
        rx.vstack(
            rx.text(
                f"Environmental Analysis: {State.business_id}",
                color=colors["white"],
                font_size="2.5em",
                margin_bottom="1em",
            ),
            rx.hstack(
                rx.box(
                    rx.text(
                        "Results:",
                        color=colors["accent_blue"],
                        font_weight="bold",
                        font_size="1.2em",
                    ),
                    rx.cond(
                        State.saved_figure,
                        rx.vstack(
                            pyplot(
                                State.pie_maker,
                                width="100%",
                                height="height",
                            ),
                            rx.text(
                                State.sentiment_summary,
                                color="black",
                                font_size="1em",
                                margin_top="0.25em",
                                text_align="center",
                            ),
                            width="100%",
                        ),
                        rx.text("Figure should go here"),
                    ),
                    background=colors["input_bg"],
                    padding="1.5em",
                    border_radius="0.5em",
                    width="52%",
                    height="500px",
                    border=f"1px solid {colors['accent_blue']}",
                    overflow="auto",
                ),
                rx.box(
                    rx.text(
                        "Next Steps:",
                        color=colors["accent_blue"],
                        font_weight="bold",
                        font_size="1.2em",
                    ),
                    rx.vstack(
                        rx.foreach(
                            State.next_steps,
                            lambda step, i: rx.hstack(
                                rx.text(
                                    font_weight="bold",
                                    margin_right="0.5em",
                                    color="black"
                                ),
                                rx.text(step, color="black"),
                                width="100%",
                                align_items="flex-start",
                                margin_bottom="0.5em",
                            ),
                        ),
                        align_items="flex-start",
                        width="100%",
                    ),
                    background=colors["input_bg"],
                    padding="1.5em",
                    border_radius="0.5em",
                    width="52%",
                    height="500px",
                    border=f"1px solid {colors['accent_blue']}",
                    overflow="auto",
                ),
                width="100%",
                justify_content="space-between",
                margin_top="1em",
            ),
            rx.button(
                "Back",
                on_click=State.go_back,
                background=colors["accent_green"],
                color=colors["text"],
                padding="0.5em 2em",
                border_radius="0.3em",
                margin_top="2em",
                class_name="hover-button",
            ),
            width="90%",
            max_width="1000px",
            align_items="center",
            padding="2em",
            margin="0 auto",
        ),
        style={
            "background": colors["light_green"],
            "minHeight": "100vh",
            "padding": "20px",
        }
    )






def index():
    return rx.box(
        rx.html(button_hover_effects),
        rx.html(f"<style>{custom_css}</style>"),
        header(),
        rx.cond(
            State.show_about_us,
            about_us_page(),
            rx.cond(
                State.show_results,
                results_page(),
                home_page(),
            ),
        ),
        width="100%",
        height="100vh",
        overflow="hidden",
    )



app = rx.App()
app.add_page(index)
