# app/services/deep_research.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
import re

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.services.web_search import retrieve_web_content
from app.services.ai import get_model
from app.core.config import settings

# Create logger
logger = logging.getLogger(__name__)


async def extract_text_from_search_result(result: Any) -> str:
    """
    Extract text content from search result, handling different types of results.
    """
    try:
        # If it's a string, return it directly
        if isinstance(result, str):
            return result
            
        # If it has a text attribute (like StringPromptValue)
        if hasattr(result, 'text'):
            return result.text
            
        # If it's a dictionary with content
        if isinstance(result, dict):
            if 'content' in result:
                return result['content']
            if 'text' in result:
                return result['text']
                
        # If it's a list of strings, join them
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            return "\n".join(result)
            
        # Fall back to string representation
        return str(result)
        
    except Exception as e:
        logger.error(f"Error extracting text from search result: {str(e)}")
        return str(result)# app/services/deep_research.py


async def generate_search_queries(query: str) -> List[str]:
    """
    Generate up to four precise search queries based on the user's query.
    """
    logger.info(f"Generating search queries for: {query}")
    
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    
    messages = [
        SystemMessage(content="You are a helpful and precise research assistant."),
        HumanMessage(content=f"User Query: {query}\n\n{prompt}")
    ]
    
    try:
        # Get AI model
        model = get_model()
        response = await model.ainvoke(messages)
        
        if response and response.content:
            # Try to extract list using regex for safer parsing
            content = response.content.strip()
            list_pattern = r'\[.*?\]'
            list_match = re.search(list_pattern, content, re.DOTALL)
            
            if list_match:
                try:
                    search_queries = eval(list_match.group(0))
                    if isinstance(search_queries, list):
                        logger.info(f"Generated {len(search_queries)} search queries")
                        return search_queries
                except Exception as e:
                    logger.error(f"Error parsing matched list: {str(e)}")
            
            # Try to extract anything that looks like search terms
            # If regex fails, try a more lenient approach - look for quotes
            query_pattern = r'[\'"]([^\'"]+)[\'"]'
            query_matches = re.findall(query_pattern, content)
            
            if query_matches:
                logger.info(f"Extracted {len(query_matches)} search queries using quote matching")
                return query_matches
            
            logger.warning(f"Failed to parse search queries from: {content}")
        
        # Fallback if parsing fails
        logger.info("Using original query as fallback")
        return [query]
    except Exception as e:
        logger.error(f"Error generating search queries: {str(e)}")
        return [query]

async def parse_web_content_results(search_results: Any) -> Dict[str, List[str]]:
    """
    Parse the search results into a structured format with urls and highlights.
    This function is specifically designed to handle the format returned by retrieve_web_content.
    """
    logger.info("Parsing web content results")
    
    # Initialize result structure
    result = {
        "urls": [],
        "highlights": []
    }
    
    # If already in correct format, return as is
    if isinstance(search_results, dict) and "urls" in search_results and "highlights" in search_results:
        return search_results
    
    try:
        # Convert results to string if it's a list or other object
        results_str = ""
        
        if isinstance(search_results, list):
            # Handle list of objects
            for item in search_results:
                # If it has a text attribute (like StringPromptValue)
                if hasattr(item, 'text'):
                    results_str += item.text
                # If it can be converted to string
                elif hasattr(item, '__str__'):
                    results_str += str(item)
                # If it's a string
                elif isinstance(item, str):
                    results_str += item
        else:
            # If not a list, convert to string
            results_str = str(search_results)
        
        # Extract URLs
        url_pattern = r'<url>(.*?)</url>'
        urls = re.findall(url_pattern, results_str)
        result["urls"] = urls
        
        # Extract highlights
        highlights_pattern = r'<highlights>(.*?)</highlights>'
        highlight_matches = re.findall(highlights_pattern, results_str, re.DOTALL)
        
        for highlight_match in highlight_matches:
            # Extract content between quotes
            quote_pattern = r'[\'"]([^\'"]+)[\'"]'
            quotes = re.findall(quote_pattern, highlight_match)
            result["highlights"].extend(quotes)
        
        # If no highlights found with the above method, try a different approach
        if not result["highlights"]:
            # Look for content inside brackets
            bracket_pattern = r'\[(.*?)\]'
            bracket_matches = re.findall(bracket_pattern, results_str, re.DOTALL)
            
            for bracket_match in bracket_matches:
                # Extract content between quotes inside brackets
                quote_pattern = r'[\'"]([^\'"]+)[\'"]'
                quotes = re.findall(quote_pattern, bracket_match)
                result["highlights"].extend(quotes)
        
        logger.info(f"Parsed {len(result['urls'])} URLs and {len(result['highlights'])} highlights")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing web content results: {str(e)}")
        return result

async def is_page_useful(query: str, page_content: Any) -> bool:
    """
    Ask the LLM if the provided webpage content is useful for answering the user's query.
    """
    logger.info("Evaluating page usefulness")
    
    try:
        # Convert the page content to a string
        text_content = await extract_text_from_search_result(page_content)
            
        prompt = (
            "You are a critical research evaluator. Given the user's query and the content of a webpage, "
            "determine if the webpage contains information relevant and useful for addressing the query. "
            "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. "
            "Do not include any extra text."
        )
        
        messages = [
            SystemMessage(content="You are a strict and concise evaluator of research relevance."),
            HumanMessage(content=f"User Query: {query}\n\nWebpage Content (first 5000 characters):\n{text_content[:5000]}\n\n{prompt}")
        ]
        
        # Get AI model
        model = get_model()
        response = await model.ainvoke(messages)
        
        if response and response.content:
            answer = response.content.strip().lower()
            return "yes" in answer
        
    except Exception as e:
        logger.error(f"Error evaluating page usefulness: {str(e)}")
    
    return False

async def extract_relevant_context(query: str, search_query: str, page_content: Any) -> str:
    """
    Extract relevant information from page content for answering the query.
    """
    logger.info("Extracting relevant context")
    
    try:
        # If page_content is already a string, use it directly
        if isinstance(page_content, str):
            text_content = page_content
        else:
            # Otherwise, convert to string
            text_content = await extract_text_from_search_result(page_content)
        
        # If text content is too short, return it directly
        if len(text_content) < 200:
            return text_content
            
        prompt = (
            "You are an expert information extractor. Given the user's query, the search query that led to this page, "
            "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
            "Return only the relevant context as plain text without commentary."
        )
        
        messages = [
            SystemMessage(content="You are an expert in extracting and summarizing relevant information."),
            HumanMessage(content=f"User Query: {query}\nSearch Query: {search_query}\n\nWebpage Content (first 5000 characters):\n{text_content[:5000]}\n\n{prompt}")
        ]
        
        # Get AI model
        model = get_model()
        response = await model.ainvoke(messages)
        
        if response and hasattr(response, 'content') and response.content:
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        elif isinstance(response, dict) and 'content' in response:
            return response['content'].strip()
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            # Return original content as fallback
            return text_content
        
    except Exception as e:
        logger.error(f"Error extracting relevant context: {str(e)}")
        # Return original content in case of error
        if isinstance(page_content, str):
            return page_content
        return ""

async def get_new_search_queries(query: str, previous_queries: List[str], all_contexts: List[str]) -> List[str]:
    """
    Determine if additional search queries are needed based on gathered information.
    """
    logger.info("Determining if more search queries are needed")
    
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with an empty string."
        "\nOutput only a Python list or the empty string without any additional text."
    )
    
    messages = [
        SystemMessage(content="You are a systematic research planner."),
        HumanMessage(content=f"User Query: {query}\nPrevious Search Queries: {previous_queries}\n\nExtracted Relevant Contexts:\n{context_combined[:10000]}\n\n{prompt}")
    ]
    
    try:
        # Get AI model
        model = get_model()
        response = await model.ainvoke(messages)
        
        if response and response.content:
            content = response.content.strip()
            
            # If empty or just whitespace, no more queries needed
            if not content or content.isspace():
                return []
            
            # Try to extract list using regex for safer parsing
            list_pattern = r'\[.*?\]'
            list_match = re.search(list_pattern, content, re.DOTALL)
            
            if list_match:
                try:
                    new_queries = eval(list_match.group(0))
                    if isinstance(new_queries, list):
                        logger.info(f"Generated {len(new_queries)} new search queries")
                        return new_queries
                except Exception as e:
                    logger.error(f"Error parsing matched list for new queries: {str(e)}")
            
            # Look for quotes as a fallback
            query_pattern = r'[\'"]([^\'"]+)[\'"]'
            query_matches = re.findall(query_pattern, content)
            
            if query_matches:
                logger.info(f"Extracted {len(query_matches)} new search queries using quote matching")
                return query_matches
            
            logger.warning(f"Failed to parse new search queries from: {content}")
        
        return []
    except Exception as e:
        logger.error(f"Error getting new search queries: {str(e)}")
        return []

async def generate_final_report(query: str, all_contexts: List[Any]) -> str:
    """
    Generate the final comprehensive report using all gathered contexts.
    """
    logger.info("Generating final report")
    
    if not all_contexts:
        logger.warning("No contexts available for report generation")
        return f"Unable to generate a comprehensive report on '{query}'. The research process did not find sufficient relevant information."
    
    # Clean and format contexts
    formatted_contexts = []
    for i, context in enumerate(all_contexts):
        try:
            if isinstance(context, str):
                formatted_contexts.append(f"Context {i+1}:\n{context}")
            elif isinstance(context, list):
                # Handle list of strings or objects
                list_content = []
                for item in context:
                    if isinstance(item, str):
                        list_content.append(item)
                    else:
                        # Try to convert to string
                        try:
                            item_str = str(item)
                            list_content.append(item_str)
                        except:
                            pass
                formatted_contexts.append(f"Context {i+1}:\n{' '.join(list_content)}")
            else:
                # Try to convert to string
                context_str = str(context)
                formatted_contexts.append(f"Context {i+1}:\n{context_str}")
        except Exception as e:
            logger.error(f"Error formatting context {i+1}: {str(e)}")
            # Skip this context if it can't be formatted
    
    if not formatted_contexts:
        logger.warning("No contexts could be formatted for report generation")
        return f"Unable to generate a comprehensive report on '{query}'. The research process encountered issues processing the gathered information."
    
    # Join the formatted contexts with separators
    context_combined = "\n\n---\n\n".join(formatted_contexts)
    
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary. "
        "If the contexts don't directly address the query, acknowledge this and provide the most relevant information available."
    )
    
    messages = [
        SystemMessage(content="You are a skilled report writer."),
        HumanMessage(content=f"User Query: {query}\n\nGathered Relevant Contexts:\n{context_combined[:20000]}\n\n{prompt}")
    ]
    
    try:
        # Get AI model
        model = get_model()
        response = await model.ainvoke(messages)
        
        # Handle different response formats
        if isinstance(response, str):
            return response.strip()
        elif hasattr(response, 'content') and isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response, dict) and 'content' in response and isinstance(response['content'], str):
            return response['content'].strip()
        else:
            # If we can't get a string response, generate a basic report
            logger.warning(f"Unexpected response format: {type(response)}")
            basic_report = f"# Research Report: {query}\n\n"
            basic_report += "## Key Findings\n\n"
            
            # Include the first 100 characters of each context as findings
            for i, context in enumerate(formatted_contexts):
                if i < 10:  # Limit to 10 findings for readability
                    lines = context.split('\n')
                    if len(lines) > 1:
                        title = lines[0]
                        content = '\n'.join(lines[1:])
                        # Get the first 100 chars of content
                        summary = content[:100] + "..." if len(content) > 100 else content
                        basic_report += f"### {title}\n{summary}\n\n"
            
            return basic_report
    except Exception as e:
        logger.error(f"Error generating final report: {str(e)}")
        
        # Create a simple report as fallback
        fallback_report = f"# Research Findings on: {query}\n\n"
        for i, context in enumerate(formatted_contexts[:10]):  # Limit to first 10 contexts
            clean_context = context.replace("Context", "Finding")
            fallback_report += f"{clean_context}\n\n---\n\n"
        
        return fallback_report

async def process_search_results(query: str, search_query: str, search_results: Any) -> List[str]:
    """
    Process search results to extract relevant contexts.
    """
    logger.info(f"Processing search results for query: {search_query}")
    contexts = []
    
    # Check if search_results is None or empty
    if not search_results:
        logger.warning("No search results to process")
        return contexts
    
    try:
        # Parse the search results into a structured format
        parsed_results = await parse_web_content_results(search_results)
        
        # Process the highlights (main content)
        for highlight in parsed_results.get("highlights", []):
            if highlight and isinstance(highlight, str):
                # Add as a context directly
                logger.info(f"Adding context (first 100 chars): {highlight[:100]}...")
                contexts.append(highlight)
        
        # If no contexts from highlights, try to extract from raw results
        if not contexts and isinstance(search_results, list):
            for item in search_results:
                try:
                    item_text = await extract_text_from_search_result(item)
                    if item_text and len(item_text) > 20:  # Avoid very short snippets
                        contexts.append(item_text)
                except Exception as e:
                    logger.error(f"Error processing list item: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in process_search_results: {str(e)}")
        
        # Fallback: try to extract text directly
        try:
            if isinstance(search_results, list):
                for item in search_results:
                    # Try direct string conversion
                    item_str = str(item)
                    if len(item_str) > 50:  # Avoid too short strings
                        contexts.append(item_str)
            else:
                # Add the entire result as a context
                result_str = str(search_results)
                if result_str:
                    contexts.append(result_str)
        except Exception as e2:
            logger.error(f"Error in fallback processing: {str(e2)}")
    
    # Ensure all contexts are strings (sanitize output)
    sanitized_contexts = []
    for context in contexts:
        if isinstance(context, str):
            sanitized_contexts.append(context)
        else:
            try:
                sanitized_contexts.append(str(context))
            except Exception as e:
                logger.error(f"Error converting context to string: {str(e)}")
    
    logger.info(f"Extracted {len(sanitized_contexts)} contexts from search results")
    return sanitized_contexts

async def deep_research(query: str, iteration_limit: int = 3) -> Dict[str, Any]:
    """
    Perform deep research on a query using iterative search and context extraction.
    """
    logger.info(f"Starting deep research for query: {query}")
    
    # Check API keys
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    
    if not settings.EXA_API_KEY:
        raise ValueError("EXA_API_KEY is not configured")
    
    aggregated_contexts = []  # All useful contexts
    all_search_queries = []   # All used search queries
    iteration = 0
    
    # Generate initial search queries
    new_search_queries = await generate_search_queries(query)
    if not new_search_queries:
        logger.warning("No search queries were generated. Using original query.")
        new_search_queries = [query]
    
    all_search_queries.extend(new_search_queries)
    
    # Iterative research loop
    while iteration < iteration_limit:
        logger.info(f"Starting iteration {iteration + 1}/{iteration_limit}")
        iteration_contexts = []
        
        # Process each search query
        for search_query in new_search_queries:
            logger.info(f"Searching for: {search_query}")
            
            try:
                # Use the retrieve_web_content function
                try:
                    # First try invoke if available
                    if hasattr(retrieve_web_content, "invoke"):
                        search_results = retrieve_web_content.invoke(search_query)
                    else:
                        # Fall back to call
                        search_results = retrieve_web_content(search_query)
                        
                    logger.info(f"Retrieved {len(search_results) if isinstance(search_results, list) else 'single'} search result(s)")
                except Exception as e:
                    logger.error(f"Error retrieving web content: {str(e)}")
                    continue
                
                # Process the search results to extract relevant contexts
                contexts = await process_search_results(query, search_query, search_results)
                if contexts:
                    iteration_contexts.extend(contexts)
                    
            except Exception as e:
                logger.error(f"Error during web search for '{search_query}': {str(e)}")
        
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
            logger.info(f"Added {len(iteration_contexts)} new contexts in iteration {iteration + 1}")
        else:
            logger.info(f"No useful contexts found in iteration {iteration + 1}")
        
        # Determine if more search queries are needed
        new_search_queries = await get_new_search_queries(query, all_search_queries, aggregated_contexts)
        
        if not new_search_queries:
            logger.info("No further research needed")
            break
        
        all_search_queries.extend(new_search_queries)
        iteration += 1
    
    # Generate final report
    try:
        final_report = await generate_final_report(query, aggregated_contexts)
    except Exception as e:
        logger.error(f"Error in final report generation: {str(e)}")
        final_report = f"Unable to generate a comprehensive report on '{query}' due to an error: {str(e)}"
        
        # Try a simpler approach if complex report generation fails
        if aggregated_contexts:
            try:
                final_report = f"Research findings on: {query}\n\n"
                for i, context in enumerate(aggregated_contexts):
                    final_report += f"Finding {i+1}:\n{context}\n\n"
            except Exception as e2:
                logger.error(f"Error in fallback report generation: {str(e2)}")
    
    return {
        "report": final_report,
        "contexts": aggregated_contexts,
        "search_queries": all_search_queries,
        "iterations": iteration + 1
    }