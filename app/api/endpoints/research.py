# Updated research.py with LangSmith tracing
import os
import logging
import asyncio
import datetime
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.services.deep_research import deep_research
from app.services.memory import get_checkpointer
from app.core.config import settings
from app.core.langsmith_setup import create_run_and_child, end_run, trace_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Get dependencies
checkpointer = get_checkpointer()

# Models
class ResearchQuery(BaseModel):
    query: str = Field(..., description="Research query to investigate deeply")
    iteration_limit: int = Field(settings.DEFAULT_RESEARCH_ITERATIONS, description="Maximum number of research iterations", ge=1, le=settings.MAX_RESEARCH_ITERATIONS)

class ResearchStatus(BaseModel):
    research_id: str
    status: str
    progress: Optional[int] = None
    total_iterations: Optional[int] = None

class ResearchResult(BaseModel):
    research_id: str
    query: str
    report: str
    contexts: Optional[List[str]] = None
    search_queries: Optional[List[str]] = None
    iterations: Optional[int] = None

# Store for active research tasks
active_research = {}

router = APIRouter()

@router.post("/query", response_model=ResearchStatus)
async def start_research(query: ResearchQuery, background_tasks: BackgroundTasks):
    """Start a deep research on the provided query."""
    # Check if API keys are configured
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")
    
    if not settings.EXA_API_KEY:
        raise HTTPException(status_code=500, detail="EXA_API_KEY is not configured")
    
    # Generate a unique ID for this research
    research_id = str(uuid.uuid4())
    logger.info(f"Starting research {research_id} for query: {query.query}")
    
    # Create a LangSmith run for the entire research process
    ls_client, ls_run = create_run_and_child(
        name="Deep Research Process",
        run_type="chain",
        inputs={"query": query.query, "iteration_limit": query.iteration_limit}
    )
    
    # Enforce iteration limit
    iteration_limit = min(query.iteration_limit, settings.MAX_RESEARCH_ITERATIONS)
    
    # Store initial status and langsmith references
    active_research[research_id] = {
        "status": "in_progress",
        "query": query.query,
        "iteration_limit": iteration_limit,
        "progress": 0,
        "total_iterations": iteration_limit,
        "langsmith_run_id": ls_run.id if ls_run else None,
        "start_time": datetime.datetime.now().isoformat()
    }
    
    # Process in background
    def process_in_background():
        try:
            logger.info(f"Starting deep research for query: {query.query}")
            
            # Create event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Enable LangSmith tracing for this task
                if "LANGSMITH_TRACING" not in os.environ:
                    os.environ["LANGSMITH_TRACING"] = "true"
                
                # Create a child run for the actual research process
                child_ls_client, child_ls_run = None, None
                if ls_client and ls_run:
                    child_ls_client, child_ls_run = create_run_and_child(
                        name="Deep Research Execution",
                        run_type="chain",
                        inputs={"query": query.query, "iteration_limit": iteration_limit}
                    )
                
                # Run deep research
                result = loop.run_until_complete(
                    deep_research(query.query, iteration_limit)
                )
                
                # Update status
                active_research[research_id] = {
                    "status": "completed",
                    "query": query.query,
                    "result": result,
                    "progress": iteration_limit,
                    "total_iterations": iteration_limit,
                    "langsmith_run_id": ls_run.id if ls_run else None,
                    "completion_time": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Research {research_id} completed successfully")
                
                # Complete the child run in LangSmith
                if child_ls_client and child_ls_run:
                    end_run(child_ls_client, child_ls_run, {
                        "report_length": len(result["report"]),
                        "contexts_count": len(result.get("contexts", [])),
                        "search_queries_count": len(result.get("search_queries", [])),
                        "iterations": result.get("iterations", 0)
                    })
                
                # Save to checkpointer for persistence
                try:
                    config = {"configurable": {"research_id": research_id}}
                    metadata = {
                        "query": query.query,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "type": "research"
                    }
                    new_versions = [{
                        "version": "1.0",
                        "timestamp": datetime.datetime.now().isoformat()
                    }]
                    checkpointer.put(config, active_research[research_id], metadata, new_versions)
                    logger.info(f"Successfully saved research result to checkpointer")
                except Exception as e:
                    logger.error(f"Error saving research result: {str(e)}")
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in deep research: {str(e)}")
            # Update status on error
            active_research[research_id] = {
                "status": "error",
                "query": query.query,
                "error": str(e),
                "progress": 0,
                "total_iterations": iteration_limit,
                "langsmith_run_id": ls_run.id if ls_run else None,
                "error_time": datetime.datetime.now().isoformat()
            }
            
            # Record error in LangSmith
            if child_ls_client and child_ls_run:
                end_run(child_ls_client, child_ls_run, error=str(e))
        
        # Complete the parent run in LangSmith
        if ls_client and ls_run:
            status = active_research[research_id]["status"]
            if status == "error":
                end_run(ls_client, ls_run, error=active_research[research_id].get("error"))
            else:
                end_run(ls_client, ls_run, {
                    "status": status,
                    "research_id": research_id,
                })
    
    background_tasks.add_task(process_in_background)
    
    # Return the research ID and initial status
    return ResearchStatus(
        research_id=research_id,
        status="in_progress",
        progress=0,
        total_iterations=iteration_limit
    )

@router.get("/status/{research_id}", response_model=ResearchStatus)
async def check_research_status(research_id: str):
    """Check the status of a research task."""
    # Try to get from active research first
    if research_id in active_research:
        research = active_research[research_id]
        return ResearchStatus(
            research_id=research_id,
            status=research["status"],
            progress=research.get("progress", 0),
            total_iterations=research.get("total_iterations", 0)
        )
    
    # Try to get from checkpointer if not active
    try:
        config = {"configurable": {"research_id": research_id}}
        research = checkpointer.get(config)
        if research:
            return ResearchStatus(
                research_id=research_id,
                status=research["status"],
                progress=research.get("progress", 0),
                total_iterations=research.get("total_iterations", 0)
            )
    except Exception as e:
        logger.error(f"Error retrieving research status: {str(e)}")
    
    # Not found
    raise HTTPException(status_code=404, detail="Research not found")

@router.get("/result/{research_id}", response_model=ResearchResult)
async def get_research_result(research_id: str):
    """Get the result of a completed research task."""
    # Try to get from active research first
    if research_id in active_research:
        research = active_research[research_id]
        if research["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Research is still {research['status']}")
        
        result = research["result"]
        return ResearchResult(
            research_id=research_id,
            query=research["query"],
            report=result["report"],
            contexts=result.get("contexts"),
            search_queries=result.get("search_queries"),
            iterations=result.get("iterations")
        )
    
    # Try to get from checkpointer if not active
    try:
        config = {"configurable": {"research_id": research_id}}
        research = checkpointer.get(config)
        if research:
            if research["status"] != "completed":
                raise HTTPException(status_code=400, detail=f"Research is still {research['status']}")
            
            result = research["result"]
            return ResearchResult(
                research_id=research_id,
                query=research["query"],
                report=result["report"],
                contexts=result.get("contexts"),
                search_queries=result.get("search_queries"),
                iterations=result.get("iterations")
            )
    except Exception as e:
        logger.error(f"Error retrieving research result: {str(e)}")
    
    # Not found
    raise HTTPException(status_code=404, detail="Research not found")

@router.get("/debug/latest", response_model=dict)
async def get_latest_research():
    """Debug endpoint to get the most recent research report."""
    if not active_research:
        return {"status": "No active research found"}
    
    # Get the most recent research ID
    latest_id = list(active_research.keys())[-1]
    latest_research = active_research[latest_id]
    
    # Check if research is completed
    if latest_research["status"] != "completed":
        return {
            "research_id": latest_id,
            "status": latest_research["status"],
            "query": latest_research["query"],
            "progress": latest_research.get("progress", 0),
            "total_iterations": latest_research.get("total_iterations", 0),
            "langsmith_run_id": latest_research.get("langsmith_run_id"),
            "message": "Research not yet completed"
        }
    
    # Return the report details
    return {
        "research_id": latest_id,
        "query": latest_research["query"],
        "status": latest_research["status"],
        "iterations": latest_research["result"].get("iterations", 0),
        "search_queries": latest_research["result"].get("search_queries", []),
        "langsmith_run_id": latest_research.get("langsmith_run_id"),
        "report_preview": latest_research["result"]["report"][:1000] + "..." if len(latest_research["result"]["report"]) > 1000 else latest_research["result"]["report"]
    }

@router.get("/debug/all", response_model=list)
async def list_all_research():
    """Debug endpoint to list all active research jobs."""
    research_list = []
    
    for research_id, research in active_research.items():
        research_info = {
            "research_id": research_id,
            "status": research["status"],
            "query": research["query"],
            "progress": research.get("progress", 0),
            "total_iterations": research.get("total_iterations", 0),
            "langsmith_run_id": research.get("langsmith_run_id")
        }
        
        # Add report info if completed
        if research["status"] == "completed" and "result" in research:
            research_info["report_length"] = len(research["result"]["report"]) if "report" in research["result"] else 0
            research_info["search_queries_count"] = len(research["result"].get("search_queries", []))
            
        research_list.append(research_info)
    
    return research_list

@router.get("/debug/report/{research_id}")
async def get_research_report(research_id: str):
    """Debug endpoint to get a research report by ID."""
    # Check active research first
    if research_id in active_research:
        research = active_research[research_id]
        if research["status"] == "completed" and "result" in research and "report" in research["result"]:
            return {
                "report": research["result"]["report"],
                "langsmith_run_id": research.get("langsmith_run_id")
            }
        else:
            return {
                "status": research["status"],
                "message": "Research not completed or report not available",
                "langsmith_run_id": research.get("langsmith_run_id")
            }
    
    # Try checkpointer as backup
    try:
        config = {"configurable": {"research_id": research_id}}
        saved_research = checkpointer.get(config)
        if saved_research and saved_research.get("status") == "completed" and "result" in saved_research:
            return {
                "report": saved_research["result"]["report"],
                "langsmith_run_id": saved_research.get("langsmith_run_id")
            }
    except Exception as e:
        return {"error": f"Error retrieving research from checkpointer: {str(e)}"}
    
    return {"error": "Research not found"}