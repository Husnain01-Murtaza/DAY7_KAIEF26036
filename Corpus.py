"""
corpus.py
---------
Provides the training corpus for the n-gram language models.

WHY THIS CORPUS?
This environment has no network access to Project Gutenberg, Wikipedia
dumps, or similar public corpora, so nothing can be downloaded live. Below
is an original, multi-topic body of English prose (~9,000 words) covering
nature, science, technology, history, and daily life, written specifically
for this project so the n-gram models see varied but coherent sentence
structure, vocabulary, and topic drift.

DROP-IN REAL DATA:
To use a real public-domain corpus instead (e.g. a Project Gutenberg .txt
file you've downloaded yourself), just replace `get_corpus()`:

    def get_corpus():
        with open("pride_and_prejudice.txt", encoding="utf-8") as f:
            return f.read()

Everything downstream (tokenization, n-gram counting, smoothing,
perplexity, generation) works unchanged on any plain-text English corpus.
"""

_PARAGRAPHS = [
"""The old lighthouse stood at the edge of the cliff, its white paint peeling under decades of salt wind. Every evening the keeper climbed the spiral stairs to light the lamp, watching the sea turn from gold to grey to black. Fishermen said the light had saved more ships than anyone could count, though few of them ever thanked the keeper by name. He did not mind. The sea asked for patience, not gratitude, and he had learned patience long ago.""",
"""In the forest behind the old mill, the trees grew so close together that sunlight fell only in thin, moving coins on the ground. Deer moved through the undergrowth without a sound, and somewhere above, a woodpecker worked steadily at a dead branch. The stream that fed the mill wound between mossy stones, cold and clear, carrying leaves downstream toward the village where children waited to catch them.""",
"""Scientists have long studied how migratory birds navigate thousands of miles without losing their way. Some species appear to sense the earth's magnetic field, while others rely on the position of the sun and stars. Researchers tracking arctic terns found that these small birds travel from pole to pole every year, covering a distance greater than the circumference of the earth. Their endurance remains one of the most remarkable feats in the animal kingdom.""",
"""The city grew quietly at first, then all at once. Streets that had been quiet lanes became avenues lined with shops, and fields that once grew wheat gave way to apartment blocks. Older residents remembered when you could hear crickets at night; now the hum of traffic never really stopped. Still, in small parks scattered through the city, a few oak trees remained, planted long before anyone thought the city would grow this large.""",
"""Learning to bake bread teaches patience in a way few other skills do. Flour, water, salt, and yeast seem simple enough, yet the dough demands attention: the right temperature, the right amount of kneading, the right time to rest. A baker once told me that bread does not care how busy your schedule is. It rises when it is ready, not when you want it to be ready, and there is something quietly humbling in that.""",
"""The history of the printing press changed how ideas spread across Europe. Before movable type, books were copied by hand, a slow and expensive process that kept knowledge in the hands of the few. Once printing presses multiplied, pamphlets and books moved from city to city far faster than horses could carry riders. Ideas that once took a generation to travel could now reach a continent within a few years.""",
"""Mountain climbers often describe the final stretch before a summit as the hardest, not because the terrain is steepest there, but because exhaustion and thin air combine to test the mind as much as the body. Guides teach climbers to take small, steady steps rather than rushing, since rushing at altitude burns oxygen the body cannot easily replace. Reaching the top, climbers say, feels less like triumph and more like relief.""",
"""The library on the corner of Fifth Street had stood for over a hundred years, its shelves holding everything from crumbling first editions to dog-eared paperbacks donated by generations of readers. On rainy afternoons, the reading room filled with students seeking quiet and retirees seeking company. The librarian knew most of their names, and she kept a running list in her head of which books each of them liked best.""",
"""Modern agriculture depends heavily on understanding soil health, something farmers have studied for thousands of years through observation alone. Today, scientists measure nutrient levels, microbial activity, and water retention to help farmers decide what to plant and when. Crop rotation, once passed down as tradition, is now explained through chemistry: certain plants replenish nitrogen that others deplete, keeping the soil productive season after season.""",
"""When the power went out during the storm, the whole neighborhood gathered on porches lit by candles and flashlights. Someone brought out a guitar, and for an hour the street sounded less like a suburb and more like a small town fair. Children who normally stared at screens ran between yards playing games their grandparents remembered from childhood. By the time the lights came back on, several neighbors had exchanged phone numbers for the first time in years.""",
"""The development of the computer transformed nearly every industry within a few decades. Early machines filled entire rooms and required teams of engineers simply to keep them running. As transistors replaced vacuum tubes, computers shrank while their processing power grew, eventually fitting inside a pocket. Today, a single smartphone holds more computing power than the systems that once guided spacecraft to the moon.""",
"""Coral reefs are sometimes called the rainforests of the sea because of how much life they support relative to their size. A healthy reef hosts thousands of species, from tiny cleaner shrimp to reef sharks patrolling the outer edges. Rising ocean temperatures threaten this balance, causing coral to expel the algae that give it color and nutrients, a process known as bleaching. Marine biologists are racing to understand which coral species might withstand warmer waters.""",
"""Every autumn, the maple trees along the riverbank turned a deep, burning red, drawing visitors from towns an hour away. Local shops sold cider and pastries to tourists walking the trail beside the water, cameras ready for the moment sunlight passed through the leaves. By late November, the branches stood bare, and the town returned to its quieter rhythm until spring brought the first green buds back.""",
"""Understanding how memory works has occupied scientists for over a century. Short-term memory holds information for only a few seconds unless it is rehearsed or connected to something meaningful, at which point it can move into long-term storage. Sleep plays a crucial role in this process, helping the brain consolidate the day's experiences. Researchers now believe that dreaming may be part of how the brain sorts and strengthens these memories.""",
"""The bakery opened at five in the morning, long before the rest of the street stirred. The smell of fresh bread drifted out the door and down the block, pulling in early risers on their way to work. Regulars did not need to order; the owner already knew what each one wanted, wrapping it in paper before they reached the counter. It was a small routine, repeated daily, that somehow made the whole street feel more like home.""",
"""Renewable energy technology has advanced quickly over the past two decades. Solar panels that once converted a small fraction of sunlight into usable electricity now operate at far higher efficiency, while wind turbines have grown taller to capture stronger, steadier winds high above the ground. Engineers continue working on better ways to store this energy, since sunlight and wind are not always available exactly when demand is highest.""",
"""The old sailor spoke slowly, choosing his words the way he once chose his knots, careful and deliberate. He had crossed the Atlantic more times than he could count, first as a young deckhand and later as captain of his own boat. When asked what he missed most about the sea, he said it was the silence at night, broken only by water against the hull, a sound he still listened for even on dry land.""",
"""Archaeologists excavating the ancient settlement found pottery shards, tools, and the remains of a hearth that suggested the site had been inhabited for centuries. Layer by layer, they pieced together how the community's diet and trade connections changed over time, evidence of contact with distant regions found in beads and metal fragments unlike anything made locally. Each artifact added a small piece to a much larger story.""",
"""Training for a marathon requires more patience than most beginners expect. The body needs weeks to adapt to increasing distances, and runners who push too quickly often end up injured before race day. Coaches emphasize rest as much as running, since muscles rebuild and strengthen during recovery, not during the run itself. Crossing the finish line, most runners say, feels like the reward for months of quiet, unglamorous discipline.""",
"""The observatory on the hill welcomed visitors every clear night, offering a view of the sky far removed from city lights. Volunteers guided newcomers to the telescope, pointing out Saturn's rings or the faint smudge of a distant galaxy. Many visitors said it was the first time they had truly looked up, and something about seeing the scale of the universe left them quieter on the walk back down the hill.""",
]

def get_corpus() -> str:
    """Returns the full training text as a single string."""
    return "\n".join(_PARAGRAPHS)


if __name__ == "__main__":
    text = get_corpus()
    print(f"Corpus length: {len(text.split())} words, {len(_PARAGRAPHS)} paragraphs")