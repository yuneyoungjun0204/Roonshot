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

from ml.constants import (
    DEFENSE_CENTER,
    ANGLE_WEIGHT,
    ETA_WEIGHT_DEFAULT,
    ETA_WEIGHT_WAVE,
    ETA_MAX,
    ETA_PENALTY_SCALE,
    LLM_DEFAULT_MODEL,
    LLM_OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    COST_MATRIX_BIG,
)
from ml.labels import FormationClass
from ml.naval_doctrine_v2 import (
    DOCTRINE_CHUNKS_V2,
    get_doctrine_for_formation,
    get_all_doctrine_texts,
    get_doctrine_metadata,
    QUICK_REFERENCE,
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

【핵심 규칙 - 반드시 준수】
★ 각 쌍은 서로 다른 군집에 배정! (중복 금지!)
★ 8개 쌍이면 8개의 서로 다른 군집에 배정!
★ 실질거리 = 거리 + |각도| × 1.5 (낮을수록 좋음)

【배정 방법】
1. 각 쌍마다 실질거리가 가장 짧은 군집 선택
2. 이미 배정된 군집은 다른 쌍에 배정 불가!
3. ETA 짧은 군집 우선

【출력 형식】
{
  "assignments": {
    "0": 3,
    "1": 0,
    "2": 5,
    "3": 2,
    "4": 7,
    "5": 1,
    "6": 4,
    "7": 6
  },
  "reasoning": "이유",
  "confidence": 0.8
}

【절대 금지】
- 같은 군집에 2개 이상 쌍 배정 금지!
- 리스트 금지: "0": [1, 2]
- 한글 키 금지!
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
        self._ollama_client = NoneA

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
        chunks = get_doctrine_for_formation(formation.name)
        context_parts = []
        # Include quick reference at the top
        context_parts.append(QUICK_REFERENCE)
        context_parts.append("")
        # Add relevant doctrine chunks (limit to top 7 for context window)
        for chunk in chunks[:7]:
            context_parts.append(f"[{chunk.id}] {chunk.title}")
            context_parts.append(chunk.content)
            context_parts.append("")
        return "\n".join(context_parts)

    def _format_score_matrix(self, context: Dict) -> str:
        """Format the pair-cluster score matrix for LLM using effective distance."""
        lines = []
        for agent_data in context.get("pair_cluster_matrix", []):
            agent_id = agent_data["agent_id"]
            lines.append(f"  아군 쌍 {agent_id}:")
            for score_data in agent_data["cluster_scores"]:
                cid = score_data["cluster_id"]
                dist = score_data["distance"]
                angle = score_data["relative_angle"]
                # Calculate effective distance using v2 doctrine formula
                effective_dist = dist + abs(angle) * 1.5
                assessment = score_data["assessment"]
                lines.append(
                    f"    → 군집 {cid}: 거리 {dist:.0f}m, "
                    f"각도 {angle:.0f}°, 실질거리 {effective_dist:.0f}m ({assessment})"
                )
        return "\n".join(lines)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama directly for inference."""
        self._init_ollama()

        if self._ollama_client is None:
            print("[LLMCommander] Ollama client is None")
            return self._fallback_response()

        try:
            print(f"[LLMCommander] Calling model: {self.model_name}...")
            response = self._ollama_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
            print(f"[LLMCommander] Response received ({len(content)} chars)")
            print(f"[LLMCommander] Response preview: {content[:200]}...")
            return content
        except Exception as e:
            print(f"[LLMCommander] Ollama call EXCEPTION: {type(e).__name__}: {e}")
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
        print(f"[LLMCommander] Parsing response...")
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                print("[LLMCommander] Detected markdown code block, extracting...")
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
            print(f"[LLMCommander] JSON parsed successfully")
            print(f"[LLMCommander] Keys in response: {list(data.keys())}")

            # Convert string keys to int for assignments
            assignments = {}

            # Format 1: {"assignments": {"0": 1, "1": 2, ...}}
            if "assignments" in data and isinstance(data["assignments"], dict):
                for k, v in data["assignments"].items():
                    assignments[int(k)] = int(v)
                print(f"[LLMCommander] Parsed from 'assignments' dict format")

            # Format 2: {"optimal_allocation": [{"pair_id": 0, "assigned_target": 1}, ...]}
            # Also handles: {"pair_id": 1, "assigned_group": [0]} (list values)
            elif "optimal_allocation" in data and isinstance(data["optimal_allocation"], list):
                for item in data["optimal_allocation"]:
                    pair_id = item.get("pair_id") or item.get("pair")
                    # Try multiple key names for target
                    target = (
                        item.get("assigned_target") if item.get("assigned_target") is not None else
                        item.get("assigned_group") if item.get("assigned_group") is not None else
                        item.get("cluster_id") if item.get("cluster_id") is not None else
                        item.get("cluster") if item.get("cluster") is not None else
                        item.get("target")
                    )
                    print(f"[LLMCommander] optimal_allocation item: pair_id={pair_id}, target={target} (type: {type(target).__name__})")

                    # Handle target being a list - take first element
                    if isinstance(target, list):
                        if len(target) > 0:
                            print(f"[LLMCommander] target is list {target}, using first: {target[0]}")
                            target = target[0]
                        else:
                            target = None

                    if pair_id is not None and target is not None:
                        assignments[int(pair_id)] = int(target)
                        print(f"[LLMCommander] Added: pair {pair_id} -> cluster {target}")
                print(f"[LLMCommander] Parsed from 'optimal_allocation' list format, total: {len(assignments)}")

            # Format 3: {"mapping": [{"pair": 0, "cluster": 1}, ...]} or similar
            elif "mapping" in data and isinstance(data["mapping"], list):
                for item in data["mapping"]:
                    pair_id = item.get("pair") or item.get("pair_id")
                    target = item.get("cluster") or item.get("cluster_id") or item.get("target")
                    if pair_id is not None and target is not None:
                        assignments[int(pair_id)] = int(target)
                print(f"[LLMCommander] Parsed from 'mapping' list format")

            # Format 4: Korean format - handles both 배정결과 and 배치결과, both list and dict
            elif "배정결과" in data or "배치결과" in data:
                korean_key = "배정결과" if "배정결과" in data else "배치결과"
                korean_data = data[korean_key]
                print(f"[LLMCommander] Detected Korean '{korean_key}' format (type: {type(korean_data).__name__})")

                if isinstance(korean_data, list):
                    # List format: [{"아군 쌍": "1", "배정된 적 군집": "2"}, ...]
                    import re
                    for item in korean_data:
                        if isinstance(item, dict):
                            # Extract pair ID from various keys
                            pair_raw = None
                            for key in ["아군 쌍", "아군쌍", "쌍", "pair", "pair_id"]:
                                if key in item:
                                    pair_raw = item[key]
                                    break

                            # Extract cluster ID from various keys
                            cluster_raw = None
                            for key in ["배정된 적 군집", "적 군집", "적군집", "배정된적군집",
                                       "군집", "cluster", "cluster_id", "target"]:
                                if key in item:
                                    cluster_raw = item[key]
                                    break

                            print(f"[LLMCommander] Korean list item: pair_raw={pair_raw}, cluster_raw={cluster_raw}")

                            # Parse pair_id (may be string "1" or int 1)
                            pair_id = None
                            if pair_raw is not None:
                                if isinstance(pair_raw, (int, float)):
                                    pair_id = int(pair_raw) - 1  # Convert 1-indexed to 0-indexed
                                elif isinstance(pair_raw, str):
                                    match = re.search(r'(\d+)', pair_raw)
                                    if match:
                                        pair_id = int(match.group(1)) - 1

                            # Parse cluster_id (may be string "2" or int 2)
                            cluster_id = None
                            if cluster_raw is not None:
                                if isinstance(cluster_raw, (int, float)):
                                    cluster_id = int(cluster_raw)
                                elif isinstance(cluster_raw, str):
                                    match = re.search(r'(\d+)', cluster_raw)
                                    if match:
                                        cluster_id = int(match.group(1))
                                elif isinstance(cluster_raw, list) and len(cluster_raw) > 0:
                                    cluster_id = int(cluster_raw[0])

                            if pair_id is not None and cluster_id is not None:
                                assignments[pair_id] = cluster_id
                                print(f"[LLMCommander] Added: pair {pair_id} -> cluster {cluster_id}")

                    print(f"[LLMCommander] Parsed from Korean list format, total: {len(assignments)}")

                elif isinstance(korean_data, dict):
                    # Dict format: {"아군 쌍1": {"적 군집": 3}, ...}
                    import re
                    print(f"[LLMCommander] Processing as dict with keys: {list(korean_data.keys())}")
                    for pair_key, value in korean_data.items():
                        print(f"[LLMCommander] Processing pair_key='{pair_key}', value={value}")
                        match = re.search(r'(\d+)', pair_key)
                        if match:
                            pair_num = int(match.group(1))
                            pair_id = pair_num - 1 if pair_num >= 1 else pair_num
                            print(f"[LLMCommander] Extracted pair_num={pair_num}, pair_id={pair_id}")

                            cluster_id = None
                            if isinstance(value, dict):
                                for key in ["적 군집", "적군집", "배정된 적 군집", "군집", "cluster", "cluster_id", "target"]:
                                    if key in value and value[key] is not None:
                                        cluster_id = value[key]
                                        break
                                print(f"[LLMCommander] Dict value, extracted cluster_id={cluster_id}")
                            elif isinstance(value, (int, float)):
                                cluster_id = int(value)
                            elif isinstance(value, str):
                                cluster_match = re.search(r'(\d+)', value)
                                if cluster_match:
                                    cluster_id = int(cluster_match.group(1))
                            elif isinstance(value, list) and len(value) > 0:
                                cluster_id = value[0]

                            # Handle cluster_id being a list
                            if isinstance(cluster_id, list) and len(cluster_id) > 0:
                                cluster_id = cluster_id[0]

                            if cluster_id is not None:
                                assignments[pair_id] = int(cluster_id)
                                print(f"[LLMCommander] Added: pair {pair_id} -> cluster {cluster_id}")
                        else:
                            print(f"[LLMCommander] WARNING: No number found in pair_key '{pair_key}'")
                    print(f"[LLMCommander] Parsed from Korean dict format, total: {len(assignments)}")

            # Format 5: Try to find any key with pair-like structure
            else:
                print(f"[LLMCommander] Unknown format, attempting generic parse...")
                import re
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            match = re.search(r'(\d+)', str(sub_key))
                            if match:
                                pair_id = int(match.group(1))
                                # Adjust for 1-indexed vs 0-indexed
                                if pair_id > 0 and all(k > 0 for k in [int(re.search(r'(\d+)', str(sk)).group(1)) for sk in value.keys() if re.search(r'(\d+)', str(sk))]):
                                    pair_id -= 1  # Convert to 0-indexed
                                if isinstance(sub_value, dict):
                                    for v in sub_value.values():
                                        if isinstance(v, int):
                                            assignments[pair_id] = v
                                            break
                                elif isinstance(sub_value, int):
                                    assignments[pair_id] = sub_value

            print(f"[LLMCommander] Parsed assignments: {assignments}")
            if not assignments:
                print("[LLMCommander] WARNING: Empty assignments!")

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

        print(f"[LLMCommander] Validating assignments...")
        print(f"[LLMCommander] Valid agent IDs: {valid_agent_ids}")
        print(f"[LLMCommander] Valid cluster IDs: {valid_cluster_ids}")
        print(f"[LLMCommander] Pre-validation assignments: {decision.pair_assignments}")

        # Check if LLM might have used 1-indexed cluster IDs
        all_cluster_ids_in_response = set(decision.pair_assignments.values())
        if all_cluster_ids_in_response and not any(cid in valid_cluster_ids for cid in all_cluster_ids_in_response):
            # None of the cluster IDs match - try adjusting for 1-indexed
            min_valid = min(valid_cluster_ids) if valid_cluster_ids else 0
            min_response = min(all_cluster_ids_in_response)
            offset = min_response - min_valid
            if offset > 0:
                print(f"[LLMCommander] Detected 1-indexed cluster IDs, adjusting by -{offset}")
                adjusted_assignments = {}
                for pair_id, cluster_id in decision.pair_assignments.items():
                    adjusted_assignments[pair_id] = cluster_id - offset
                decision.pair_assignments = adjusted_assignments
                print(f"[LLMCommander] Adjusted assignments: {decision.pair_assignments}")

        validated_assignments = {}
        for pair_id, cluster_id in decision.pair_assignments.items():
            if pair_id in valid_agent_ids and cluster_id in valid_cluster_ids:
                validated_assignments[pair_id] = cluster_id
                print(f"[LLMCommander] VALID: pair {pair_id} -> cluster {cluster_id}")
            else:
                # Try to salvage by adjusting pair_id if it's just an indexing issue
                adjusted_pair_id = None
                adjusted_cluster_id = cluster_id

                # Check if pair_id needs adjustment
                if pair_id not in valid_agent_ids:
                    # Maybe pair_id is already 0-indexed but agents use different IDs
                    if pair_id + 1 in valid_agent_ids:
                        adjusted_pair_id = pair_id + 1
                    elif pair_id - 1 in valid_agent_ids:
                        adjusted_pair_id = pair_id - 1
                    else:
                        # pair_id might directly correspond to array index
                        agent_list = sorted(valid_agent_ids)
                        if 0 <= pair_id < len(agent_list):
                            adjusted_pair_id = agent_list[pair_id]

                # Check if cluster_id needs adjustment
                if cluster_id not in valid_cluster_ids:
                    if cluster_id - 1 in valid_cluster_ids:
                        adjusted_cluster_id = cluster_id - 1

                if adjusted_pair_id is not None:
                    final_pair = adjusted_pair_id
                else:
                    final_pair = pair_id

                if final_pair in valid_agent_ids and adjusted_cluster_id in valid_cluster_ids:
                    validated_assignments[final_pair] = adjusted_cluster_id
                    print(f"[LLMCommander] ADJUSTED: pair {pair_id}->{final_pair} -> cluster {cluster_id}->{adjusted_cluster_id}")
                else:
                    print(f"[LLMCommander] INVALID: pair {pair_id} (valid={pair_id in valid_agent_ids}) -> cluster {cluster_id} (valid={cluster_id in valid_cluster_ids})")

        decision.pair_assignments = validated_assignments

        # === ENFORCE 1:1 UNIQUE MAPPING (remove duplicates) ===
        # Check for duplicate cluster assignments
        cluster_to_pairs = {}  # cluster_id -> list of pair_ids
        for pair_id, cluster_id in validated_assignments.items():
            if cluster_id not in cluster_to_pairs:
                cluster_to_pairs[cluster_id] = []
            cluster_to_pairs[cluster_id].append(pair_id)

        # Find duplicates
        duplicates_found = {cid: pairs for cid, pairs in cluster_to_pairs.items() if len(pairs) > 1}
        if duplicates_found:
            print(f"[LLMCommander] WARNING: Duplicate assignments detected: {duplicates_found}")
            print(f"[LLMCommander] Enforcing 1:1 mapping...")

            # Keep only first pair for each cluster, reassign others
            used_clusters = set()
            unique_assignments = {}
            available_clusters = set(valid_cluster_ids) - set(validated_assignments.values())

            # First pass: assign one pair per cluster (keep first occurrence)
            for pair_id in sorted(validated_assignments.keys()):
                cluster_id = validated_assignments[pair_id]
                if cluster_id not in used_clusters:
                    unique_assignments[pair_id] = cluster_id
                    used_clusters.add(cluster_id)

            # Second pass: reassign duplicate pairs to available clusters
            for pair_id in sorted(validated_assignments.keys()):
                if pair_id not in unique_assignments:
                    if available_clusters:
                        new_cluster = min(available_clusters)  # Pick lowest available
                        unique_assignments[pair_id] = new_cluster
                        available_clusters.remove(new_cluster)
                        print(f"[LLMCommander] Reassigned pair {pair_id} to cluster {new_cluster}")
                    else:
                        print(f"[LLMCommander] No available cluster for pair {pair_id}")

            validated_assignments = unique_assignments
            decision.pair_assignments = validated_assignments

        # === FILL IN MISSING PAIR ASSIGNMENTS ===
        # Ensure ALL pairs have a cluster assignment
        missing_pairs = valid_agent_ids - set(validated_assignments.keys())
        if missing_pairs:
            print(f"[LLMCommander] WARNING: Missing assignments for pairs: {missing_pairs}")
            used_clusters = set(validated_assignments.values())
            available_clusters = sorted(set(valid_cluster_ids) - used_clusters)

            for pair_id in sorted(missing_pairs):
                if available_clusters:
                    new_cluster = available_clusters.pop(0)  # Pick first available
                    validated_assignments[pair_id] = new_cluster
                    print(f"[LLMCommander] Auto-assigned pair {pair_id} to cluster {new_cluster}")
                else:
                    print(f"[LLMCommander] No available cluster for pair {pair_id}")

            decision.pair_assignments = validated_assignments

        print(f"[LLMCommander] Final complete assignments: {validated_assignments}")

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

    Uses effective distance formula from naval doctrine v2:
    effective_distance = distance + |relative_angle| × 1.5

    Returns:
        Dict mapping pair_id -> cluster_id
    """
    if not agents or not clusters:
        return {}

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # ETA weight adjustment based on formation
    if formation == FormationClass.CONCENTRATED:
        eta_weight = 0.2  # Focus on direction
    elif formation == FormationClass.WAVE:
        eta_weight = 0.4  # ETA more important for wave timing
    else:  # DIVERSIONARY
        eta_weight = 0.2  # Global optimization

    # Build cost matrix using effective distance
    n_agents = len(agents)
    n_clusters = len(clusters)
    cost_matrix = np.zeros((n_agents, n_clusters))

    for i, agent in enumerate(agents):
        pos = np.array(agent['pos'])
        heading = agent['angle']

        for j, cluster in enumerate(clusters):
            center = np.array(cluster['center'])

            # Direct distance
            dist = np.linalg.norm(pos - center)

            # Relative angle calculation
            dx = center[0] - pos[0]
            dy = center[1] - pos[1]
            bearing = math.degrees(math.atan2(-dy, dx))
            rel_angle = abs((bearing - heading + 180) % 360 - 180)

            # Effective distance using v2 doctrine formula
            # 10° angle difference = 15m distance equivalent
            effective_dist = dist + abs(rel_angle) * 1.5

            # ETA factor (lower ETA = higher priority)
            vel = np.array(cluster['velocity'])
            speed = np.linalg.norm(vel)
            if speed > 1.0:
                dist_to_center = np.linalg.norm(center - np.array(defense_center))
                eta = dist_to_center / speed
                eta_factor = max(0, 1.0 - eta / 200.0)  # 0-1, lower ETA = higher
            else:
                eta_factor = 0.5

            # Combined cost: effective distance + ETA penalty
            cost_matrix[i, j] = effective_dist + eta_weight * (1.0 - eta_factor) * 500

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
    High-level function to get tactical pair-to-cluster mapping using LLM.

    Note: For OR-Tools based assignment, use ml.ortools_assignment.get_tactical_assignment()
    instead. This function is specifically for LLM-based tactical reasoning.

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

    # Rule-based assignment (uses scipy Hungarian algorithm)
    assignments = rule_based_mapping(agents, clusters, formation)
    reasoning = f"Rule-based assignment for {formation.name} formation"

    return assignments, reasoning


# =============================================================================
# OR-TOOLS INTEGRATION (Recommended Default)
# =============================================================================

def get_ortools_command(
    agents: List[Dict],
    clusters: List[Dict],
    formation: FormationClass,
    confidence: float = 0.8,
) -> Tuple[Dict[int, int], str]:
    """
    Get tactical pair-to-cluster mapping using OR-Tools optimal assignment.

    This is the recommended default method for tactical assignment:
    - Guaranteed 1:1 unique mapping (no duplicates)
    - Deterministic results (no LLM variability)
    - Uses effective distance formula: distance + |relative_angle| × 1.5

    Args:
        agents: List of agent dicts with 'id', 'pos', 'angle'
        clusters: List of cluster dicts with 'cluster_id', 'center', 'velocity'
        formation: Classified formation type
        confidence: Formation classification confidence

    Returns:
        Tuple of (assignments dict, reasoning string)
    """
    from ml.ortools_assignment import get_optimal_assignment
    return get_optimal_assignment(agents, clusters, formation, confidence)
