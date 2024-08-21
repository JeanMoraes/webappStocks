import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# CRIANDO YAHOO FINANCE TOOL
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func = lambda ticket: fetch_stock_price(ticket)
)

# IMPORTANTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model='gpt-3.5-turbo')


stockPriceAnalyst = Agent(
    role = "Senior stock price analyst",
    goal = "Find the {ticket} stock price and analyses trends",
    backstory = """You're a highly experienced in analysing the price of an specific stock
    and make predictions about its future price""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    tools = [yahoo_finance_tool],
    allow_delegation = False
)

getStockPrice = Task(
    description = "Analyze the stock {ticket} price history ans create a trend analyses of up, down or sideways",
    expected_output = """Specify the current trend stock price - up, down or sideways.
    eg. stock = 'AAPL, price UP'""",
    agent = stockPriceAnalyst
)

# IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results = 10)

newsAnalyst = Agent(
    role = "Stock news analyst",
    goal = """Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with the news context.
    For each request stock assert, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
    """,
    backstory = """You're a highly experienced in analysing the Market trends and News and have tracked assests for more then 10 years.
    You'are also master level analyst in the tradicional Market and have deep understanding of human psychology. You understand News,
    theirs titles and informations, but you look at those with a health dose of skepticism. You conside also the source of the News articles.
    """,
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    tools = [search_tool],
    allow_delegation = False
)

get_news = Task(
    description = f"""Take the stock and Always include BTC to it (if not request).
    Use the search tool to search each one individually.
    The current date is {datetime.now()}.
    Compose the results into a helpfull report""",
    expected_output = """A summary of the overall Market and one sentence summary for each request assert.
    Include a fear/greed score for each assert based on the News. Use the format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent = newsAnalyst
)


stockAnalystWriter = Agent(
    role = "Senior stock analyst writer",
    goal = "Analyze the trends price and News and write na insightfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.",
    backstory = """You're a widely accepted as the best stock analyst in the Market. You understand complex concepts and create
    compelling stories and naratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories - eg. cycle theory fundamental analyses.
    You're able to hold multiple opinions when analyzing anything.
    """,
    verbose = True,
    llm = llm,
    max_iter = 1, #5
    memory = True,
    allow_delegation = True
)


writeAnalyses = Task(
    description = """Use the stock price trend and the stock News report to create na analyses and write the newsletter
    about the {ticket} company that is brief and highlights the most import points.
    Focus on the stock price trend, News and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and News summary.
    """,
    expected_output = """An eloquente 3 paragraph newsletter formated as markdown in an easy readble manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall Picture anb spike up the interest
    - main parte provides the meat of the analysis include the News summary and fear/greed scores.
    - summary - key facts and concrete future tren prediction - up, down or sideways""",
    agent = stockAnalystWriter,
    context = [getStockPrice, get_news]
)


crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks = [getStockPrice, get_news, writeAnalyses],
    process = Process.hierarchical,
    manager_llm = llm,
    verbose = True,
    full_output = True,
    share_crew = False,
    max_iter = 15,
)


with st.sidebar:
    st.header('Entre the ticket stock')

    with st.form(key='research_form'):
        topic = st.text_input('Select the ticket')
        submit_button = st.form_submit_button(label = 'Run Research')

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        myresults = crew.kickoff(inputs={'ticket': topic})

        st.subheader('Results of your research:')
        st.write(myresults['final_output'])
