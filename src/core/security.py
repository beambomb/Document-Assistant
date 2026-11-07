"""Role-Based Access Control (RBAC) and ACL enforcement layer."""

from typing import List, Set, Dict, Any
import logging
from src.core.models import UserContext, Chunk
from config.settings import settings

logger = logging.getLogger(__name__)


class AccessControlManager:
    """Enforces Role-Based Access Control (RBAC) on documents and chunks."""

    def __init__(self, role_hierarchy: Dict[str, List[str]] | None = None):
        self.role_hierarchy = role_hierarchy or settings.load_role_hierarchy()

    def get_user_allowed_roles(self, user: UserContext) -> Set[str]:
        """Resolves the complete set of clearance tags granted to the user."""
        allowed: Set[str] = set()
        for role in user.roles:
            role_norm = role.strip().lower()
            allowed.add(role_norm)
            if role_norm in self.role_hierarchy:
                for inherited in self.role_hierarchy[role_norm]:
                    allowed.add(inherited.strip().lower())
        return allowed

    def is_chunk_accessible(self, chunk: Chunk, user: UserContext) -> bool:
        """Determines if a user has clearance to view the chunk."""
        user_roles = self.get_user_allowed_roles(user)
        chunk_roles = {r.strip().lower() for r in chunk.access_roles}

        # If chunk is marked 'public', anyone can access
        if "public" in chunk_roles:
            return True

        # Check intersection between chunk requirements and user roles
        return bool(chunk_roles.intersection(user_roles))

    def filter_chunks(self, chunks: List[Chunk], user: UserContext) -> List[Chunk]:
        """Filters a list of chunks, retaining only those the user is cleared for."""
        allowed_roles = self.get_user_allowed_roles(user)
        cleared_chunks: List[Chunk] = []

        for chunk in chunks:
            chunk_roles = {r.strip().lower() for r in chunk.access_roles}
            if "public" in chunk_roles or chunk_roles.intersection(allowed_roles):
                cleared_chunks.append(chunk)
            else:
                logger.debug(
                    f"ACL Block: User {user.user_id} with roles {user.roles} denied access "
                    f"to chunk {chunk.chunk_id} requiring {chunk.access_roles}"
                )

        return cleared_chunks

    def get_filter_condition(self, user: UserContext) -> Dict[str, Any]:
        """Generates metadata filter dictionary for vector databases (Chroma/LanceDB/Qdrant)."""
        allowed = list(self.get_user_allowed_roles(user))
        # Suitable for ChromaDB: {"$or": [{"access_roles": {"$contains": role}} for role in allowed]}
        return {"allowed_roles": allowed}


# Global security manager instance
access_manager = AccessControlManager()
