import streamlit as st
from agent import ResearchAgent, ImagesAgent, VideosAgent

def main():
    # Define the agents
    serpapi_key = "cff1141144fc78b9790b84e240e0cdf7f42b6021b9075698c179dda0abd464ce"
    aiml_api_key = "6f1030c3d57e40a1b5261e8d1c52da6a "
    aiml_base_url = "https://api.aimlapi.com"

    research_agent = ResearchAgent(
        role="research agent",
        task="conduct various types of searches and gather information give a overal review on it",
        backstory="You are detailed and efficient in gathering information.",
        tools=["SerpAPI", "AIML API"],
        serpapi_key=serpapi_key,
        aiml_api_key=aiml_api_key,
        aiml_base_url=aiml_base_url
    )

    images_agent = ImagesAgent(
        role="images agent",
        task="search and display images",
        backstory="You are efficient in finding and showing relevant images.",
        tools=["SerpAPI"],
        serpapi_key=serpapi_key,
        aiml_api_key=aiml_api_key,
        aiml_base_url=aiml_base_url
    )

    videos_agent = VideosAgent(
        role="videos agent",
        task="search and display videos",
        backstory="You are efficient in finding and showing relevant videos.",
        tools=["SerpAPI"],
        serpapi_key=serpapi_key,
        aiml_api_key=aiml_api_key,
        aiml_base_url=aiml_base_url
    )

    # Streamlit UI
    st.title("Research and Media Agent")

    query = st.text_input("Enter search query:", "Coffee")
    location = st.text_input("Enter location (optional):", "Austin, TX, Texas, United States")

    if st.button("Search"):
        st.header("Top 5 Search Results with Summary")
        search_results = research_agent.perform_search(query)
        summary = research_agent.summarize_results(search_results)
        st.write("Summary:")
        st.write(summary)
        
        st.write("Detailed Results:")
        for result in search_results[:5]:
            st.write(f"**{result['title']}**")
            st.write(result["link"])
            st.write(result["snippet"])

        st.header("Top 10 Related Images")
        image_results = images_agent.perform_image_search(query, location)
        for image in image_results[:10]:
            st.image(image["thumbnail"], caption=image["title"], use_column_width=True)

        st.header("Top 10 Related Videos")
        video_results = videos_agent.perform_video_search(query)
        if video_results:
            for video in video_results[:10]:
                st.video(video["link"], caption=video.get("title", "No title available"))
        else:
            st.write("No videos found.")

   

    

if __name__ == "__main__":
    main()
