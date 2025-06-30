# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!
# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        """This document is about the history of the band Queen. Queen formed in London in 1970 and quickly rose to fame with their theatrical sound and ambitious songwriting. The band began when guitarist Brian May and drummer Roger Taylor, previously in the band Smile, were joined by vocalist Freddie Mercury and later bassist John Deacon. Their early work, including Queen (1973) and Queen II (1974), laid the groundwork for what would become their signature blend of rock, opera, and glam. Their 1975 album A Night at the Opera, featuring the iconic “Bohemian Rhapsody,” marked a major breakthrough and established them as global stars. Over the next decade, Queen released hit after hit, including “We Are the Champions,” “Another One Bites the Dust,” and “Under Pressure.” Their Live Aid performance in 1985 is considered one of the greatest live shows in rock history. After Mercury’s death in 1991, the band’s activity slowed, though they later resumed touring with other vocalists. Queen's music continues to captivate audiences around the world, decades after their debut. Their combination of innovation, charisma, and powerful performance established them as one of the most enduring rock bands of all time.""",
        
        """This document is about the Queen's band members. Queen’s lineup featured four unique musicians who each played vital roles in shaping the band’s identity. Freddie Mercury, born Farrokh Bulsara in 1946, was the band’s dynamic lead singer and a principal songwriter. Known for his incredible vocal range and theatrical stage presence, Mercury wrote classics like “Killer Queen” and “Don’t Stop Me Now.” Brian May, Queen’s guitarist, is recognized for his distinctive tone and complex arrangements, crafted with his homemade “Red Special” guitar. He wrote iconic tracks such as “We Will Rock You” and “The Show Must Go On.” Roger Taylor, the drummer, added high-energy rhythms and backing vocals, contributing hits like “Radio Ga Ga.” John Deacon, the bassist, was a reserved yet essential presence, writing “Another One Bites the Dust” and “I Want to Break Free.” The members shared songwriting duties, a rare dynamic that gave their discography wide musical variety. After Mercury passed away in 1991, Deacon retired, while May and Taylor continued to perform under the Queen name with new collaborators. Their collective creativity and chemistry defined Queen’s lasting success.""",
        
        """This document is about Queen's discography. Queen's discography showcases the band’s musical evolution and creative ambition. Their debut album, Queen (1973), introduced them as a force in hard rock, while Queen II (1974) displayed complex arrangements and fantasy themes. With Sheer Heart Attack (1974), they gained mainstream attention, but it was A Night at the Opera (1975) that catapulted them to global fame, featuring the groundbreaking single “Bohemian Rhapsody.” The band continued their success with A Day at the Races (1976), News of the World (1977), and The Game (1980), which featured “Another One Bites the Dust.” In the 1980s, they experimented with electronic sounds in albums like Hot Space (1982) and returned to rock in The Works (1984). Their final studio album with Mercury, Innuendo (1991), was followed by Made in Heaven (1995), completed using Mercury’s final recordings. Queen has also released numerous live albums, soundtracks, and compilation albums, including Greatest Hits, one of the best-selling albums in UK history. Their discography reflects a fearless willingness to explore different genres and redefine rock music.""",
        
        """This document talks about Queen's awards and nominations. Queen’s contributions to music have earned them numerous awards and honors throughout their career. The band was inducted into the Rock and Roll Hall of Fame in 2001 and the Songwriters Hall of Fame in 2003. “Bohemian Rhapsody” received entry into the Grammy Hall of Fame in 2004, recognizing its artistic and historical significance. Although Queen received relatively few Grammy Awards during their early years, they were honored with the Grammy Lifetime Achievement Award in 2018, cementing their status as rock legends. Queen has received Brit Awards, including for Outstanding Contribution to Music in 1990. Their influence extended to film with the 2018 biopic Bohemian Rhapsody, which won four Academy Awards, including Best Actor for Rami Malek’s portrayal of Freddie Mercury. The film also won the Golden Globe for Best Motion Picture – Drama. In addition to critical recognition, Queen's commercial success includes over 300 million records sold globally. Their awards reflect both their musical innovation and the enduring impact of their legacy across generations and genres.""",
        
        """This document explains Queen's music style and influences. Queen is celebrated for their eclectic and innovative music style, which combines elements of rock, opera, glam, pop, and funk. From the beginning, the band embraced theatricality and complexity. Songs like “Bohemian Rhapsody” seamlessly shift between ballad, opera, and hard rock, showcasing their ambition and refusal to adhere to traditional song structures. Freddie Mercury’s love for opera and musical theater influenced their dramatic vocal arrangements, while Brian May’s guitar work added melodic and harmonic depth with layered solos. Roger Taylor brought a hard rock edge with his drumming and raspy vocals, and John Deacon infused rhythm and groove influenced by funk and soul. Their ability to integrate these styles is evident across albums like A Night at the Opera, The Game, and Innuendo. Queen also made early use of studio technology for multi-track layering and effects, enhancing their sound’s richness. Their wide-ranging influences included The Beatles, Jimi Hendrix, Led Zeppelin, and classical composers. This fusion created a distinct identity that broke barriers and expanded the boundaries of rock music."""


    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
st.title("👑🎤Queen: The Rock Royalty🎸")

st.markdown("<h3 style=color:#6a0dad;'>🎶 Welcome to the Queen Knowledge Hub! 🎶</h3>", unsafe_allow_html=True)
st.markdown('<span style="color:#cd7f32;"><em>Explore the remarkable journey, groundbreaking music, and enduring legacy of one of the greatest bands in rock history.</em></span>', unsafe_allow_html=True)

# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
st.write("🎤 Welcome to my deep dive into the legendary rock band Queen! Explore their history, members, albums, awards, and the iconic sound that changed music forever.")

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
question = st.text_input("Curious about Queen? Type your question here!🎧")

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
st.markdown("""
    <style>
        .stButton>button {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)
if st.button("Rock me the answer!🎸", type="primary"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("🎤 Warming up Freddie’s mic..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
        st.success("👑 The Queen archives have spoken – enjoy your insight!")
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")
        st.info("💡 Try something like: *'What albums did Queen release in the 1980s?'*")
        

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
with st.expander("About this Queen Q&A System"):
    st.markdown("""
    <div style='color:purple; font-size:16px;'>
        <p>This system is built to answer your questions about the legendary rock band Queen, including:</p>
        <ul>
            <li>The history and rise of Queen</li>
            <li>Detailed profiles of each band member</li>
            <li>Albums, songs, and major discography highlights</li>
            <li>Awards, nominations, and career milestones</li>
            <li>Musical style, influences, and genre-blending innovations</li>
        </ul>
        <p>Dive in and discover the stories, sounds, and legacy behind one of the greatest bands in rock history.</p>
    </div>
    """, unsafe_allow_html=True)
# TO RUN: Save as app.py, then type: streamlit run app.py

