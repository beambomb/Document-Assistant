"""Unit tests for Role-Based Access Control (RBAC) and ACL pre-filtering."""

import pytest
from src.core.models import UserContext, Chunk
from src.core.security import AccessControlManager


@pytest.fixture
def acl_manager():
    return AccessControlManager()


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chk_pub_1",
            doc_id="PUB-GEN-001",
            title="Profil Perusahaan",
            text="Informasi publik tentang visi dan misi perusahaan.",
            access_roles=["public"],
        ),
        Chunk(
            chunk_id="chk_emp_1",
            doc_id="SOP-IT-001",
            title="SOP Password Laptop",
            text="Karyawan wajib mengganti password setiap 90 hari.",
            access_roles=["employee", "internal"],
        ),
        Chunk(
            chunk_id="chk_fin_1",
            doc_id="SOP-FIN-003",
            title="Reimbursement Operasional",
            text="Prosedur klaim biaya perjalanan dinas maksimal 5 juta.",
            access_roles=["finance"],
        ),
        Chunk(
            chunk_id="chk_hr_1",
            doc_id="KEB-HR-002",
            title="Prosedur Disipliner Karyawan",
            text="Sanksi SP1, SP2, dan SP3 berlaku bagi pelanggaran kehadiran.",
            access_roles=["hr"],
        ),
        Chunk(
            chunk_id="chk_sec_1",
            doc_id="KEB-SEC-005",
            title="Standar Gaji dan Bonus Direksi",
            text="Rincian kompensasi eksekutif dan persentase pembagian laba tahunan.",
            access_roles=["management", "confidential"],
        ),
    ]


def test_role_hierarchy_resolution(acl_manager):
    guest = UserContext(user_id="u_guest", roles=["guest"])
    employee = UserContext(user_id="u_emp", roles=["employee"])
    finance = UserContext(user_id="u_fin", roles=["finance"])
    mgmt = UserContext(user_id="u_mgmt", roles=["management"])

    assert acl_manager.get_user_allowed_roles(guest) == {"guest", "public"}
    assert {"public", "employee", "internal"}.issubset(acl_manager.get_user_allowed_roles(employee))
    assert {"public", "employee", "finance"}.issubset(acl_manager.get_user_allowed_roles(finance))
    assert {"public", "employee", "finance", "hr", "management", "confidential"}.issubset(
        acl_manager.get_user_allowed_roles(mgmt)
    )


def test_guest_access_isolation(acl_manager, sample_chunks):
    guest = UserContext(user_id="u_guest", roles=["guest"])
    accessible = acl_manager.filter_chunks(sample_chunks, guest)

    assert len(accessible) == 1
    assert accessible[0].chunk_id == "chk_pub_1"
    prohibited_ids = {"chk_emp_1", "chk_fin_1", "chk_hr_1", "chk_sec_1"}
    assert all(chk.chunk_id not in prohibited_ids for chk in accessible)


def test_employee_cannot_see_confidential_or_finance(acl_manager, sample_chunks):
    employee = UserContext(user_id="u_emp", roles=["employee"])
    accessible = acl_manager.filter_chunks(sample_chunks, employee)

    accessible_ids = {c.chunk_id for c in accessible}
    assert "chk_pub_1" in accessible_ids
    assert "chk_emp_1" in accessible_ids
    assert "chk_fin_1" not in accessible_ids
    assert "chk_hr_1" not in accessible_ids
    assert "chk_sec_1" not in accessible_ids


def test_management_can_access_all(acl_manager, sample_chunks):
    mgmt = UserContext(user_id="u_mgmt", roles=["management"])
    accessible = acl_manager.filter_chunks(sample_chunks, mgmt)

    assert len(accessible) == len(sample_chunks)
    assert any(c.chunk_id == "chk_sec_1" for c in accessible)


def test_zero_leakage_guarantee(acl_manager, sample_chunks):
    malicious_probe = UserContext(user_id="attacker", roles=["guest"])
    filtered = acl_manager.filter_chunks(sample_chunks, malicious_probe)

    for chunk in filtered:
        assert "confidential" not in chunk.access_roles
        assert "management" not in chunk.access_roles
        assert "hr" not in chunk.access_roles
