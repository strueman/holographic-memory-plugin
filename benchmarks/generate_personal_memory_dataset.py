"""Generate a realistic personal memory dataset for benchmarking.

Creates:
- Declarative facts (50-300 chars, personal statements)
- Realistic queries (who, what, when, where about personal info)
- Multiple hops required for some queries
- Various categories: personal, work, finance, health, preferences, events
"""

import json
import random
import sqlite3
import os
import sys

random.seed(42)

# Generate realistic personal facts
FACTS = [
    # Personal info
    "User's full name is Simon",
    "User's partner is named Rachel",
    "User has a dog named Max",
    "User has a cat named Lola",
    "User lives in Melbourne, Australia",
    "User works at Google as a software engineer",
    "User's birthday is March 15th",
    "User's email is simon@example.com",
    "User's phone number ends in 4567",
    "User drives a silver Honda Civic",
    "User bought the Honda Civic on February 10th",
    "User's car insurance is with State Farm",
    "User's insurance premium decreased by $20/month",
    "User has a credit card with 50,000 points",
    "User redeemed 50,000 points for a $500 gift card",
    "User bought car accessories with the gift card",
    
    # Work
    "User started at Google in 2024",
    "User's previous job was at NovaTech",
    "User worked at NovaTech for 3 years",
    "User's role at Google is senior software engineer",
    "User works on ML/AI projects at Google",
    "User's team has 12 engineers",
    "User's manager is named David",
    "User's team lead is named Sarah",
    "User has a standing desk at work",
    "User commutes by train for 45 minutes each way",
    "User works from home on Tuesdays",
    "User works from home on Thursdays",
    "User's work schedule is 9am to 5pm",
    "User takes lunch break at 12:30pm",
    "User's desk is in building 41",
    
    # Finance
    "User's savings account balance is $15,000",
    "User invests in index funds",
    "User has a Roth IRA account",
    "User's rent is $2,500/month",
    "User's car payment is $350/month",
    "User's internet bill is $80/month",
    "User's phone bill is $65/month",
    "User's gym membership is $45/month",
    "User's Netflix subscription is $15/month",
    "User's Spotify subscription is $10/month",
    "User's total monthly expenses are about $3,500",
    "User's monthly income is about $6,000",
    "User saves about $2,500 per month",
    "User's savings rate is 42%",
    "User has an emergency fund of 6 months expenses",
    
    # Health
    "User's height is 180cm",
    "User's weight is 78kg",
    "User exercises 4 times per week",
    "User runs 5km on Mondays",
    "User goes to the gym on Wednesdays",
    "User does yoga on Fridays",
    "User swims on Sundays",
    "User's resting heart rate is 58 bpm",
    "User's blood pressure is 120/80",
    "User takes vitamin D supplements daily",
    "User's cholesterol level is 180 mg/dL",
    "User's blood sugar is normal",
    "User drinks 2 liters of water per day",
    "User sleeps 7-8 hours per night",
    "User's bedtime is 10:30pm",
    "User wakes up at 6:00am on weekdays",
    
    # Preferences
    "User prefers dark mode on all devices",
    "User's favorite programming language is Python",
    "User uses VS Code as primary editor",
    "User prefers mechanical keyboards",
    "User's favorite coffee is pour-over",
    "User drinks 3 cups of coffee per day",
    "User's favorite restaurant is a local Italian place",
    "User prefers tea over coffee in the evening",
    "User's favorite book genre is science fiction",
    "User's favorite movie is Inception",
    "User prefers Linux over Windows",
    "User uses Debian as primary OS",
    "User's favorite music genre is electronic",
    "User listens to podcasts during commute",
    "User's favorite podcast is Lex Fridman",
    "User prefers audiobooks over physical books",
    "User's favorite author is Neal Stephenson",
    "User prefers cats over dogs",
    "User's favorite food is sushi",
    "User is allergic to shellfish",
    
    # Events
    "User attended a conference in San Francisco in March",
    "User's anniversary is June 15th",
    "User's friend Emily got married in April",
    "User attended a wedding in April",
    "User's birthday party was in March",
    "User went to a BBQ event in June",
    "User attended a tech meetup last week",
    "User went to a concert in May",
    "User had dinner with Rachel on Friday",
    "User went hiking on Sunday",
    "User visited a museum two months ago",
    "User attended a cooking class last month",
    "User went to a art gallery opening in April",
    "User attended a charity event in May",
    "User had a team lunch on Wednesday",
    
    # Shopping
    "User bought running shoes from Adidas",
    "User bought a new keyboard from Keychron",
    "User bought a monitor from Dell",
    "User bought a mouse from Logitech",
    "User bought a desk lamp from IKEA",
    "User bought a plant from a local nursery",
    "User bought groceries at SaveMart",
    "User bought coffee beans from a local roaster",
    "User bought a new phone case from Amazon",
    "User bought a gift for Rachel at TK Maxx",
    "User bought a book from BookWorld",
    "User bought a gift for Max (dog) from PetSmart",
    "User bought a new mattress from SleepWell",
    "User bought a new laptop bag from Amazon",
    "User bought a new pair of headphones from BestBuy",
    
    # Travel
    "User went to San Francisco in March",
    "User went to Yellowstone in May",
    "User plans to visit Japan in September",
    "User went to Chicago in June",
    "User flew with United Airlines",
    "User stayed at a hotel in downtown San Francisco",
    "User's flight to San Francisco was on March 15th",
    "User's return flight was on March 18th",
    "User drove from hometown to Yellowstone",
    "User camped at Grant Village campground",
    "User's road trip to Yellowstone was 5 days",
    "User visited Old Faithful at Yellowstone",
    "User saw the Grand Prismatic Spring",
    "User took the Beartooth Highway",
    "User's total road trip distance was 2,500 miles",
    
    # Hobbies
    "User is learning machine learning",
    "User plays guitar on weekends",
    "User reads science fiction books",
    "User watches sci-fi movies",
    "User gardens on weekends",
    "User grows tomatoes and cucumbers",
    "User has a vegetable garden",
    "User started gardening in April",
    "User plants 10 tomato plants",
    "User plants 5 cucumber plants",
    "User bakes bread on weekends",
    "User practices photography",
    "User takes photos during hikes",
    "User edits photos in Lightroom",
    "User has a collection of vinyl records",
    "User collects mechanical keyboards",
    "User builds model cars",
    "User started building model cars in 2024",
    "User built a Ferrari model",
    "User built a Porsche 991 model",
    
    # Social
    "User's best friend is named Mark",
    "User's sister is named Sarah",
    "User's brother is named Tom",
    "User has 3 siblings total",
    "User's parents live in Sydney",
    "User calls parents on Sundays",
    "User's childhood home was in Brisbane",
    "User moved to Melbourne in 2020",
    "User met Rachel at a tech meetup",
    "User and Rachel have been together for 2 years",
    "User's colleague Alex just became a parent",
    "User's colleague Tom just became a parent",
    "User attended Alex's baby shower",
    "User attended Tom's baby shower",
    "User's neighbor is named Emily",
    
    # Technology
    "User has a MacBook Pro",
    "User has an iPhone 14 Pro",
    "User has an iPad",
    "User has an Apple Watch",
    "User uses GitHub for version control",
    "User contributes to open source projects",
    "User has 15 GitHub repositories",
    "User's GitHub username is simon-dev",
    "User uses Docker for development",
    "User uses Kubernetes for deployment",
    "User's server is at home",
    "User's server specs are Ryzen 9 8945",
    "User has 64GB RAM on server",
    "User has a 24GB GPU for inference",
    "User runs LLMs locally",
    
    # Education
    "User has a bachelor's degree in Computer Science",
    "User graduated from University of Melbourne",
    "User graduated in 2018",
    "User's GPA was 3.8/4.0",
    "User's thesis was on sentiment analysis",
    "User presented thesis poster at a conference",
    "User attended a conference in San Francisco",
    "User presented a poster on sentiment analysis",
    "User's undergrad research was at a university",
    "User's undergraduate course research was at a university",
]

# Generate queries
QUERIES = [
    # Simple lookups
    {"query": "What is the user's name?", "expected": ["User's full name is Simon"]},
    {"query": "Who is the user's partner?", "expected": ["User's partner is named Rachel"]},
    {"query": "What is the user's dog's name?", "expected": ["User has a dog named Max"]},
    {"query": "What is the user's cat's name?", "expected": ["User has a cat named Lola"]},
    {"query": "Where does the user live?", "expected": ["User lives in Melbourne, Australia"]},
    {"query": "What is the user's job?", "expected": ["User works at Google as a software engineer"]},
    {"query": "What is the user's birthday?", "expected": ["User's birthday is March 15th"]},
    {"query": "What car does the user drive?", "expected": ["User drives a silver Honda Civic"]},
    
    # Finance
    {"query": "What is the user's monthly rent?", "expected": ["User's rent is $2,500/month"]},
    {"query": "How much does the user save per month?", "expected": ["User saves about $2,500 per month"]},
    {"query": "What is the user's savings rate?", "expected": ["User's savings rate is 42%"]},
    {"query": "What is the user's phone bill?", "expected": ["User's phone bill is $65/month"]},
    {"query": "What is the user's gym membership cost?", "expected": ["User's gym membership is $45/month"]},
    
    # Work
    {"query": "When did the user start at Google?", "expected": ["User started at Google in 2024"]},
    {"query": "How long did the user work at NovaTech?", "expected": ["User worked at NovaTech for 3 years"]},
    {"query": "What is the user's role at Google?", "expected": ["User's role at Google is senior software engineer"]},
    {"query": "How long is the user's commute?", "expected": ["User commutes by train for 45 minutes each way"]},
    {"query": "What days does the user work from home?", "expected": ["User works from home on Tuesdays", "User works from home on Thursdays"]},
    
    # Health
    {"query": "How often does the user exercise?", "expected": ["User exercises 4 times per week"]},
    {"query": "What does the user do on Mondays?", "expected": ["User runs 5km on Mondays"]},
    {"query": "What does the user do on Wednesdays?", "expected": ["User goes to the gym on Wednesdays"]},
    {"query": "What time does the user wake up on weekdays?", "expected": ["User wakes up at 6:00am on weekdays"]},
    {"query": "What is the user's resting heart rate?", "expected": ["User's resting heart rate is 58 bpm"]},
    
    # Preferences
    {"query": "What is the user's favorite programming language?", "expected": ["User prefers dark mode on all devices", "User's favorite programming language is Python"]},
    {"query": "What editor does the user use?", "expected": ["User uses VS Code as primary editor"]},
    {"query": "How many cups of coffee does the user drink?", "expected": ["User drinks 3 cups of coffee per day"]},
    {"query": "What is the user's favorite food?", "expected": ["User's favorite food is sushi"]},
    {"query": "What is the user allergic to?", "expected": ["User is allergic to shellfish"]},
    
    # Events
    {"query": "Where did the user go in March?", "expected": ["User went to San Francisco in March"]},
    {"query": "Where did the user go in May?", "expected": ["User went to Yellowstone in May"]},
    {"query": "Where is the user planning to go in September?", "expected": ["User plans to visit Japan in September"]},
    {"query": "What airline does the user fly with?", "expected": ["User flew with United Airlines"]},
    {"query": "When was the user's flight to San Francisco?", "expected": ["User's flight to San Francisco was on March 15th"]},
    
    # Hobbies
    {"query": "What is the user learning?", "expected": ["User is learning machine learning"]},
    {"query": "What does the user play on weekends?", "expected": ["User plays guitar on weekends"]},
    {"query": "What does the user grow in the garden?", "expected": ["User grows tomatoes and cucumbers"]},
    {"query": "How many tomato plants does the user have?", "expected": ["User plants 10 tomato plants"]},
    {"query": "What model cars has the user built?", "expected": ["User built a Ferrari model", "User built a Porsche 991 model"]},
    
    # Social
    {"query": "Who is the user's best friend?", "expected": ["User's best friend is named Mark"]},
    {"query": "Who is the user's sister?", "expected": ["User's sister is named Sarah"]},
    {"query": "Where do the user's parents live?", "expected": ["User's parents live in Sydney"]},
    {"query": "When does the user call parents?", "expected": ["User calls parents on Sundays"]},
    {"query": "When did the user move to Melbourne?", "expected": ["User moved to Melbourne in 2020"]},
    
    # Technology
    {"query": "What laptop does the user have?", "expected": ["User has a MacBook Pro"]},
    {"query": "What phone does the user have?", "expected": ["User has an iPhone 14 Pro"]},
    {"query": "What is the user's GitHub username?", "expected": ["User's GitHub username is simon-dev"]},
    {"query": "What CPU is in the user's server?", "expected": ["User's server specs are Ryzen 9 8945"]},
    {"query": "How much RAM does the user's server have?", "expected": ["User has 64GB RAM on server"]},
    
    # Education
    {"query": "What degree does the user have?", "expected": ["User has a bachelor's degree in Computer Science"]},
    {"query": "Where did the user go to university?", "expected": ["User graduated from University of Melbourne"]},
    {"query": "What was the user's thesis topic?", "expected": ["User's thesis was on sentiment analysis"]},
    {"query": "When did the user graduate?", "expected": ["User graduated in 2018"]},
    {"query": "What was the user's GPA?", "expected": ["User's GPA was 3.8/4.0"]},
    
    # Multi-hop queries
    {"query": "What is the user's total monthly expenses?", "expected": ["User's total monthly expenses are about $3,500"]},
    {"query": "What is the user's monthly income?", "expected": ["User's monthly income is about $6,000"]},
    {"query": "What car did the user buy and when?", "expected": ["User drives a silver Honda Civic", "User bought the Honda Civic on February 10th"]},
    {"query": "What accessories did the user buy with the gift card?", "expected": ["User bought car accessories with the gift card"]},
    {"query": "What is the user's dog's name and what is the cat's name?", "expected": ["User has a dog named Max", "User has a cat named Lola"]},
    {"query": "Where does the user live and where do their parents live?", "expected": ["User lives in Melbourne, Australia", "User's parents live in Sydney"]},
    {"query": "What days does the user work from home and what time do they wake up?", "expected": ["User works from home on Tuesdays", "User works from home on Thursdays", "User wakes up at 6:00am on weekdays"]},
    {"query": "What does the user do on weekends?", "expected": ["User plays guitar on weekends", "User gardens on weekends", "User bakes bread on weekends"]},
    {"query": "What subscriptions does the user have?", "expected": ["User's Netflix subscription is $15/month", "User's Spotify subscription is $10/month"]},
    {"query": "What is the user's commute time and schedule?", "expected": ["User commutes by train for 45 minutes each way", "User's work schedule is 9am to 5pm"]},
]

def generate_dataset():
    """Generate the dataset."""
    # Create facts list
    facts = []
    for i, fact in enumerate(FACTS):
        facts.append({
            "fact_id": i + 1,
            "content": fact,
            "category": "general",
            "tags": "",
            "trust_score": 0.8,
        })
    
    # Create queries list
    queries = []
    for i, q in enumerate(QUERIES):
        # Map expected fact strings to fact_ids
        expected_ids = []
        for expected_str in q["expected"]:
            for fact in facts:
                if fact["content"] == expected_str:
                    expected_ids.append(fact["fact_id"])
                    break
        
        queries.append({
            "query_id": f"query_{i+1:03d}",
            "query": q["query"],
            "query_type": "personal-memory",
            "expected": expected_ids,
        })
    
    return facts, queries

def create_benchmark_db(facts, queries, db_path):
    """Create a benchmark SQLite database."""
    conn = sqlite3.connect(db_path)
    
    # Create tables
    conn.executescript("""
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        
        CREATE VIRTUAL TABLE facts_fts USING fts5(
            content, tags, content=facts, content_rowid=fact_id
        );
        
        CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
        END;
        
        CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
        END;
        
        CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
            INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
        END;
        
        CREATE TABLE queries (
            query_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            query_type TEXT DEFAULT 'personal-memory',
            expected TEXT NOT NULL
        );
    """)
    
    # Insert facts
    for fact in facts:
        conn.execute(
            "INSERT INTO facts (fact_id, content, category, tags, trust_score) VALUES (?, ?, ?, ?, ?)",
            (fact["fact_id"], fact["content"], fact["category"], fact["tags"], fact["trust_score"])
        )
    
    # Insert queries
    for query in queries:
        conn.execute(
            "INSERT INTO queries (query_id, query, query_type, expected) VALUES (?, ?, ?, ?)",
            (query["query_id"], query["query"], query["query_type"], json.dumps(query["expected"]))
        )
    
    conn.commit()
    conn.close()
    
    return len(facts), len(queries)

if __name__ == "__main__":
    facts, queries = generate_dataset()
    
    db_path = os.path.expanduser("~/.hermes/benchmarks/personal_memory_benchmark.db")
    num_facts, num_queries = create_benchmark_db(facts, queries, db_path)
    
    print(f"Generated {num_facts} facts and {num_queries} queries")
    print(f"Database: {db_path}")
    
    # Print sample
    print("\nSample facts:")
    for f in facts[:5]:
        print(f"  [{f['fact_id']}] {f['content']}")
    
    print("\nSample queries:")
    for q in queries[:5]:
        print(f"  {q['query']}")
        print(f"    Expected: {q['expected']}")
