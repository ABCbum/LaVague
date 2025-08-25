from lavague.contexts.openai import OpenaiContext
from lavague.core import ActionEngine, WorldModel
from lavague.core.agents import WebAgent
from lavague.core.token_counter import TokenCounter
from lavague.drivers.playwright import PlaywrightDriver
from lavague.core.navigation import NavigationEngine
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from dotenv import load_dotenv

load_dotenv()


playwright_driver = PlaywrightDriver(url="https://www.saucedemo.com/v1/inventory.html")
token_counter = TokenCounter(log=False)
context = OpenaiContext(llm="gpt-4.1", mm_llm="gpt-4.1")
action_engine = ActionEngine.from_context(
    context=context, 
    driver=playwright_driver,
    navigation_engine=NavigationEngine(playwright_driver)   
)
world_model = WorldModel.from_context(context)
agent = WebAgent(
    world_model, action_engine, n_steps=1, token_counter=token_counter
)

action_result = agent.run("Add item to shopping cart and checkout")