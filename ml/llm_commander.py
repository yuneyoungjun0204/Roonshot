"""
LLM Commander with RAG System for Tactical Pair-to-Cluster Mapping

This module integrates:
  1. Naval doctrine knowledge base (RAG retrieval)
  2. Tactical situation report (NLG)
  3. LLM-based tactical decision making
  4. Structured JSON output for pair assignments

Recommended LLM Models (via Ollama):
  - Primary: qwen2.5:7b-instruct (Best Korean + structured output)
  - Alternative: llama3.1:8b-instruct (Wide support, good reasoning)
  - Lightweight: mistral:7b-instruct (Fastest inference)
"""

import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ml.constants import DEFENSE_CENTER
from ml.labels import FormationClass
from ml.naval_doctrine import (
    DOCTRINE_CHUNKS,
    get_doctrine_by_formation,
    get_all_doctrine_texts,
    get_doctrine_metadata,
)
from ml.tactical_report import (
    prepare_llm_input,
    AgentState,
    ClusterState,
    TacticalReportGenerator,
)


@dataclass
class MappingDecision:
    """LLM's tactical mapping decision."""
    pair_assignments: Dict[int, int]  # pair_id -> cluster_id
    reasoning: str
    confidence: float
    raw_response: str


# =============================================================================
# LLM PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """당신은 해상 USV 방어 작전의 전술 지휘관입니다.
아군 USV 쌍(Pair)을 적 군집(Cluster)에 최적으로 배정하는 임무를 수행합니다.

【지휘관 역할】
1. 전술 상황 보고서를 분석합니다.
2. 해군 교범의 지침을 참조합니다.
3. 각 아군 쌍을 하나의 적 군집에 배정합니다.
4. 배정 결정의 근거를 명확히 제시합니다.

【결정 원칙】
- 거리가 가까운 쌍을 해당 군집에 우선 배정
- 상대 각도가 작은 쌍 우선 (선회 시간 최소화)
- ETA가 짧은 군집에 더 많은 전력 배정
- 모선 보호가 최우선 임무

【출력 형식】
반드시 아래 JSON 형식으로만 응답하세요:
{
  "assignments": {
    "쌍_ID": 군집_ID,
    ...
  },
  "reasoning": "배정 근거 설명",
  "confidence": 0.0~1.0 신뢰도
}
"""

RAG_CONTEXT_TEMPLATE = """【참조 교범】
{doctrine_context}
"""

QUERY_TEMPLATE = """【전술 상황 보고】
{tactical_report}

【분석 데이터】
- 포메이션: {formation}
- 아군 쌍 수: {num_agents}개
- 적 군집 수: {num_clusters}개

【매핑 점수 매트릭스】
{score_matrix}

위 상황을 분석하고, 각 아군 쌍을 적 군집에 배정하는 최적의 결정을 내려주세요.
JSON 형식으로 응답하세요."""


class LLMCommander:
    """
    LLM-based tactical commander using RAG for doctrine-informed decisions.

    Supports multiple backends:
    - Ollama (local): qwen2.5, llama3.1, mistral
    - LangChain (for advanced RAG features)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        use_langchain: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        defense_center: Tuple[float, float] = DEFENSE_CENTER,
    ):
        """
        Initialize the LLM Commander.

        Args:
            model_name: Ollama model name to use
            use_langchain: Whether to use LangChain for RAG (requires extra deps)
            ollama_base_url: Ollama server URL
            defense_center: Defense center coordinates
        """
        self.model_name = model_name
        self.use_langchain = use_langchain
        self.ollama_base_url = ollama_base_url
        self.defense_center = defense_center
        self.report_generator = TacticalReportGenerator(defense_center)

        # RAG components (initialized lazily)
        self._vector_store = None
        self._langchain_chain = None
        self._ollama_client = None

    def _init_ollama(self):
        """Initialize Ollama client."""
        if self._ollama_client is not None:
            return

        try:
            import ollama
            self._ollama_client = ollama.Client(host=self.ollama_base_url)
            # Test connection
            self._ollama_client.list()
            print(f"[LLMCommander] Ollama connected: {self.model_name}")
        except ImportError:
            print("[LLMCommander] Warning: ollama package not installed. pip install ollama")
            self._ollama_client = None
        except Exception as e:
            print(f"[LLMCommander] Warning: Ollama connection failed: {e}")
            self._ollama_client = None

    def _init_langchain_rag(self):
        """Initialize LangChain RAG system with vector store."""
        if self._langchain_chain is not None:
            return

        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_community.chat_models import ChatOllama
            from langchain.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.ollama_base_url,
            )

            # Create vector store from doctrine
            texts = get_all_doctrine_texts()
            metadatas = get_doctrine_metadata()

            self._vector_store = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=embeddings,
                collection_name="naval_doctrine",
            )

            # Create retriever
            retriever = self._vector_store.as_retriever(
                search_kwargs={"k": 5}
            )

            # Create LLM
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                format="json",
                temperature=0.1,
            )

            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", RAG_CONTEXT_TEMPLATE + "\n\n" + QUERY_TEMPLATE),
            ])

            # Create chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self._langchain_chain = (
                {
                    "doctrine_context": retriever | format_docs,
                    "tactical_report": RunnablePassthrough(),
                    "formation": RunnablePassthrough(),
                    "num_agents": RunnablePassthrough(),
                    "num_clusters": RunnablePassthrough(),
                    "score_matrix": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            print("[LLMCommander] LangChain RAG initialized")

        except ImportError as e:
            print(f"[LLMCommander] LangChain not available: {e}")
            print("Install with: pip install langchain langchain-community chromadb")
            self._langchain_chain = None

    def _get_doctrine_context(self, formation: FormationClass) -> str:
        """Get relevant doctrine text for the given formation."""
        chunks = get_doctrine_by_formation(formation.name)
        context_parts = []
        for chunk in chunks[:5]:  # Limit to top 5 relevant chunks
            context_parts.append(f"[{chunk.id}] {chunk.title}")
            context_parts.append(chunk.content)
            context_parts.append("")
        return "\n".join(context_parts)

    def _format_score_matrix(self, context: Dict) -> str:
        """Format the pair-cluster score matrix for LLM."""
        lines = []
        for agent_data in context.get("pair_cluster_matrix", []):
            agent_id = agent_data["agent_id"]
            lines.append(f"  아군 쌍 {agent_id}:")
            for score_data in agent_data["cluster_scores"]:
                cid = score_data["cluster_id"]
                dist = score_data["distance"]
                angle = score_data["relative_angle"]
                score = score_data["engagement_score"]
                assessment = score_data["assessment"]
                lines.append(
                    f"    → 군집 {cid}: 거리 {dist:.0f}m, "
                    f"각도 {angle:.0f}°, 점수 {score:.2f} ({assessment})"
                )
        return "\n".join(lines)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama directly for inference."""
        self._init_ollama()

        if self._ollama_client is None:
            return self._fallback_response()

        try:
            response = self._ollama_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"[LLMCommander] Ollama call failed: {e}")
            return self._fallback_response()

    def _fallback_response(self) -> str:
        """Generate a fallback response when LLM is unavailable."""
        return json.dumps({
            "assignments": {},
            "reasoning": "LLM unavailable - using rule-based assignment",
            "confidence": 0.5,
        })

    def _parse_llm_response(self, response: str) -> MappingDecision:
        """Parse LLM JSON response into MappingDecision."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Handle markdown code blocks
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```json"):
                        in_json = True
                        continue
                    if line.startswith("```"):
                        in_json = False
                        continue
                    if in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            data = json.loads(response)

            # Convert string keys to int for assignments
            assignments = {}
            for k, v in data.get("assignments", {}).items():
                assignments[int(k)] = int(v)

            return MappingDecision(
                pair_assignments=assignments,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLMCommander] Failed to parse response: {e}")
            print(f"[LLMCommander] Raw response: {response[:500]}")
            return MappingDecision(
                pair_assignments={},
                reasoning=f"Parse error: {e}",
                confidence=0.0,
                raw_response=response,
            )

    def get_mapping_decision(
        self,
        agents: List[Dict],
        clusters: List[Dict],
        formation: FormationClass,
        confidence: float = 0.8,
    ) -> MappingDecision:
        """
        Get tactical mapping decision from LLM.

        Args:
            agents: List of agent dicts with 'id', 'pos', 'angle'
            clusters: List of cluster dicts with 'cluster_id', 'center', 'count', 'velocity'
            formation: Classified formation type
            confidence: Formation classification confidence

        Returns:
            MappingDecision with pair-to-cluster assignments
        """
        # Generate tactical report and context
        report, context = prepare_llm_input(
            agents, clusters, formation, confidence, self.defense_center
        )

        # Get doctrine context
        doctrine_context = self._get_doctrine_context(formation)

        # Format score matrix
        score_matrix = self._format_score_matrix(context)

        # Build prompt
        prompt = RAG_CONTEXT_TEMPLATE.format(doctrine_context=doctrine_context)
        prompt += "\n\n"
        prompt += QUERY_TEMPLATE.format(
            tactical_report=report,
            formation=formation.name,
            num_agents=len(agents),
            num_clusters=len(clusters),
            score_matrix=score_matrix,
        )

        # Call LLM
        if self.use_langchain:
            self._init_langchain_rag()
            if self._langchain_chain is not None:
                # Use LangChain RAG chain
                response = self._langchain_chain.invoke({
                    "tactical_report": report,
                    "formation": formation.name,
                    "num_agents": len(agents),
                    "num_clusters": len(clusters),
                    "score_matrix": score_matrix,
                })
            else:
                response = self._call_ollama(prompt)
        else:
            response = self._call_ollama(prompt)

        # Parse and return decision
        decision = self._parse_llm_response(response)

        # Validate assignments
        valid_agent_ids = {a['id'] for a in agents}
        valid_cluster_ids = {c['cluster_id'] for c in clusters}

        validated_assignments = {}
        for pair_id, cluster_id in decision.pair_assignments.items():
            if pair_id in valid_agent_ids and cluster_id in valid_cluster_ids:
                validated_assignments[pair_id] = cluster_id

        decision.pair_assignments = validated_assignments

        return decision


# =============================================================================
# RULE-BASED FALLBACK (when LLM is unavailable)
# =============================================================================

def rule_based_mapping(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass,
    defense_center: Tuple[float, float] = DEFENSE_CENTER,
) -> Dict[int, int]:
    """
    Rule-based pair-to-cluster mapping as fallback.

    Uses the same logic as the existing Hungarian algorithm but
    with formation-specific adjustments.

    Returns:
        Dict mapping pair_id -> cluster_id
    """
    if not agents or not clusters:
        return {}

    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # Get weight adjustments based on formation
    if formation == FormationClass.CONCENTRATED:
        w_dist, w_angle, w_eta = 0.3, 0.5, 0.2
    elif formation == FormationClass.WAVE:
        w_dist, w_angle, w_eta = 0.3, 0.3, 0.4
    else:  # DIVERSIONARY
        w_dist, w_angle, w_eta = 0.5, 0.3, 0.2

    # Build cost matrix
    n_agents = len(agents)
    n_clusters = len(clusters)
    cost_matrix = np.zeros((n_agents, n_clusters))

    for i, agent in enumerate(agents):
        pos = np.array(agent['pos'])
        heading = agent['angle']

        for j, cluster in enumerate(clusters):
            center = np.array(cluster['center'])

            # Distance cost
            dist = np.linalg.norm(pos - center)
            dist_cost = dist / 10000.0  # Normalize

            # Angle cost
            dx = center[0] - pos[0]
            dy = center[1] - pos[1]
            bearing = math.degrees(math.atan2(-dy, dx))
            rel_angle = abs((bearing - heading + 180) % 360 - 180)
            angle_cost = rel_angle / 180.0  # Normalize

            # ETA cost (lower ETA = higher priority = lower cost)
            vel = np.array(cluster['velocity'])
            speed = np.linalg.norm(vel)
            if speed > 1.0:
                dist_to_center = np.linalg.norm(center - np.array(defense_center))
                eta = dist_to_center / speed
                eta_cost = 1.0 - min(eta / 200.0, 1.0)  # Invert: low ETA = low cost
            else:
                eta_cost = 0.5

            # Combined cost
            cost_matrix[i, j] = w_dist * dist_cost + w_angle * angle_cost + w_eta * eta_cost

    # Pad to square if needed
    size = max(n_agents, n_clusters)
    padded_cost = np.full((size, size), 1e9)
    padded_cost[:n_agents, :n_clusters] = cost_matrix

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost)

    # Build assignment map
    assignments = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_agents and c < n_clusters:
            pair_id = agents[r]['id']
            cluster_id = clusters[c]['cluster_id']
            assignments[pair_id] = cluster_id

    return assignments


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================

def get_tactical_command(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass,
    confidence: float = 0.8,
    use_llm: bool = True,
    model_name: str = "qwen2.5:7b-instruct",
) -> Tuple[Dict[int, int], str]:
    """
    High-level function to get tactical pair-to-cluster mapping.

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'count', 'velocity'
        formation: Classified formation type
        confidence: Formation classification confidence
        use_llm: Whether to use LLM (falls back to rule-based if False or LLM fails)
        model_name: Ollama model to use

    Returns:
        Tuple of (assignments dict, reasoning string)
    """
    if use_llm:
        commander = LLMCommander(model_name=model_name)
        decision = commander.get_mapping_decision(agents, clusters, formation, confidence)

        if decision.pair_assignments:
            return decision.pair_assignments, decision.reasoning

        # LLM failed, fall back to rule-based
        print("[get_tactical_command] LLM failed, using rule-based fallback")

    # Rule-based assignment
    assignments = rule_based_mapping(agents, clusters, formation)
    reasoning = f"Rule-based assignment for {formation.name} formation"

    return assignments, reasoning
