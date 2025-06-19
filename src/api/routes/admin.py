# src/api/routes/admin.py
from fastapi import APIRouter, HTTPException, status, Depends, Request, Body
from datetime import datetime
import logging
from bson import ObjectId
from typing import List, Dict, Any, Optional
import os
import uuid
from pydantic import BaseModel, EmailStr, Field
import traceback

from api.routes.auth import get_current_user, get_password_hash, encrypt_db_password, generate_company_code
from api.database import get_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/admin", tags=["admin"])

# First admin initialization model
class InitAdminCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50) 
    password: str = Field(..., min_length=8)
    full_name: str
    init_key: str  # Secret initialization key

# Company creation model
class CompanyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    admin_email: Optional[EmailStr] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    industry: Optional[str] = None
    issabel_db_host: Optional[str] = None
    issabel_db_user: Optional[str] = None
    issabel_db_password: Optional[str] = None
    issabel_db_name: Optional[str] = None
    issabel_db_port: Optional[int] = 3306

# Admin endpoint to create the first admin user (no auth required)
@router.post("/initialize-admin", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def initialize_admin(admin: InitAdminCreate = Body(...), db = Depends(get_database)):
    """
    Create the first admin account during system initialization.
    This requires a special initialization key and should only be used once
    when setting up the system initially.
    """
    response = {
        "success": False,
        "errors": [],
        "data": None
    }
    
    try:
        # Verify initialization key from environment variable
        expected_key = os.getenv("ADMIN_INIT_KEY")
        if not expected_key:
            logger.error("ADMIN_INIT_KEY environment variable not set")
            response["errors"].append("System not properly configured for admin initialization")
            return response
        
        if admin.init_key != expected_key:
            logger.warning(f"Attempt to initialize admin with invalid key from IP: {admin.email}")
            response["errors"].append("Invalid initialization key")
            return response
        
        # Check if any admin users already exist
        existing_admin = await db["users"].find_one({"is_admin": True})
        if existing_admin:
            logger.warning(f"Attempt to initialize admin when admins already exist: {admin.email}")
            response["errors"].append("Admin users already exist. Use the promote-admin endpoint instead.")
            return response
        
        # Check if email already exists
        if await db["users"].find_one({"email": admin.email}):
            response["errors"].append("Email already registered")
            return response
        
      
        
        # Create admin user
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Hash password
        try:
            hashed_password = get_password_hash(admin.password)
        except Exception as e:
            logger.error(f"Password hashing error: {str(e)}")
            response["errors"].append(f"Error processing password: {str(e)}")
            return response
        
        # Create initial admin user data
        user_data = {
            "_id": user_id,
            "email": admin.email,
            "username": admin.username,
            "hashed_password": hashed_password,
            "full_name": admin.full_name,
            "company_id": "None",
            "company_code": "None",
            "company_name": "None",
            "created_at": created_at,
            "is_active": True,
            "is_verified": True,
            "is_admin": True,
            "system_initialized": True,
            "initialization_date": created_at
        }
        
        # Insert into MongoDB
        result = await db["users"].insert_one(user_data)
        logger.info(f"🔑 Created first admin user: {admin.username}")
        
        response["success"] = True
        response["data"] = {
            "message": "First administrator account created successfully",
            "id": user_id,
            "email": admin.email,
            "username": admin.username,
            "is_admin": True,
            "created_at": created_at
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        logger.error(traceback.format_exc())
        response["errors"].append(f"Failed to create admin user: {str(e)}")
        return response

# Admin endpoint to verify users
@router.post("/verify-user", status_code=status.HTTP_200_OK)
async def admin_verify_user(email: str, company_code: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to verify a user and assign them to a company"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can verify users"
        )
    
    # Find the user to verify by email
    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user is already verified
    if user.get("is_verified", False):
        return {
            "message": "User is already verified",
            "email": email,
            "username": user.get("username"),
            "is_verified": True
        }
    
    # Validate company code
    company = await db["companies"].find_one({"company_code": company_code})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company code"
        )
    
    # Update user with company info and verify
    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "company_id": str(company["_id"]),
                "company_code": company_code,
                "company_name": company["name"],
                "is_verified": True,
                "verified_by": current_user["username"],
                "verified_at": datetime.utcnow()
            }
        }
    )
    
    logger.info(f"User '{email}' verified by admin '{current_user['username']}' and assigned to company '{company['name']}'")
    
    return {
        "message": "User verified successfully",
        "email": email,
        "username": user.get("username"),
        "company_name": company["name"],
        "company_code": company_code,
        "is_verified": True
    }

# Admin endpoint to create or promote a user to admin
@router.post("/promote-admin", status_code=status.HTTP_200_OK)
async def promote_to_admin(email: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to promote a user to administrator status"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only existing administrators can promote users to admin status"
        )
    
    # Find the user to promote by email
    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user is already an admin
    if user.get("is_admin", False):
        return {
            "message": "User is already an administrator",
            "email": email,
            "username": user.get("username"),
            "is_admin": True
        }
    
    # Update user to admin status
    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "is_admin": True,
                "promoted_by": current_user["username"],
                "promoted_at": datetime.utcnow(),
                "is_verified": True  # Ensure admin users are always verified
            }
        }
    )
    
    logger.info(f"User '{email}' promoted to admin by '{current_user['username']}'")
    
    return {
        "message": "User promoted to administrator successfully",
        "email": email,
        "username": user.get("username"),
        "is_admin": True
    }

# Admin endpoint to get list of pending verifications
@router.get("/pending-verifications", status_code=status.HTTP_200_OK)
async def get_pending_verifications(current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Get list of users pending verification"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can view pending verifications"
        )
    
    # Find all users pending verification
    pending_users = await db["users"].find(
        {
            "is_verified": False,
            "verification_requested": True
        },
        {
            "_id": 1,
            "username": 1,
            "email": 1,
            "full_name": 1,
            "verification_requested_at": 1,
            "created_at": 1
        }
    ).to_list(length=100)  # Limit to 100 results
    
    # Format response
    formatted_users = []
    for user in pending_users:
        # Calculate time pending in hours
        requested_at = user.get("verification_requested_at", user.get("created_at"))
        time_pending = None
        if requested_at:
            delta = datetime.utcnow() - requested_at
            time_pending = round(delta.total_seconds() / 3600, 1)  # Hours with 1 decimal place
        
        formatted_users.append({
            "id": str(user["_id"]),
            "email": user["email"],
            "username": user["username"],
            "full_name": user.get("full_name"),
            "requested_at": requested_at,
            "time_pending_hours": time_pending
        })
    
    # Sort by time pending (descending)
    formatted_users.sort(key=lambda x: x.get("time_pending_hours", 0), reverse=True)
    
    return {
        "count": len(formatted_users),
        "pending_users": formatted_users
    }

# Endpoint to update user company information
@router.post("/update-user-company", status_code=status.HTTP_200_OK)
async def update_user_company(email: str, company_code: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to update a user's company code and automatically sync the company name"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can update user company information"
        )
    
    # Find the user by email
    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Handle the case where admin wants to remove company association
    if not company_code or company_code.lower() == "none":
        await db["users"].update_one(
            {"_id": user["_id"]},
            {"$set": {
                "company_id": "None",
                "company_code": "None",
                "company_name": "None"
            }}
        )
        
        logger.info(f"Admin '{current_user['username']}' removed company association for user '{email}'")
        
        return {
            "message": "User company information cleared successfully",
            "email": email,
            "company_name": "None",
            "company_code": "None"
        }
    
    # Otherwise, verify and update with new company code
    company = await db["companies"].find_one({"company_code": company_code})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company code"
        )
    
    # Update user with company info
    await db["users"].update_one(
        {"_id": user["_id"]},
        {"$set": {
            "company_id": str(company["_id"]),
            "company_code": company_code,
            "company_name": company["name"],
            "updated_by": current_user["username"],
            "updated_at": datetime.utcnow()
        }}
    )
    
    logger.info(f"Admin '{current_user['username']}' updated company for user '{email}' to '{company['name']}' ({company_code})")
    
    return {
        "message": "User company information updated successfully",
        "email": email,
        "company_name": company["name"],
        "company_code": company_code,
        "previous_company_code": user.get("company_code"),
        "previous_company_name": user.get("company_name")
    }

# Admin endpoint to sync company names
@router.post("/sync-company-names", status_code=status.HTTP_200_OK)
async def sync_company_names(current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """
    Admin endpoint to synchronize company names for all users with the correct names from the companies collection.
    This fixes any cases where a user has a company_code but the company_name is empty, 'None', or incorrect.
    """
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can perform this operation"
        )
    
    # Find all users who have a company_code but potentially incorrect company_name
    users_to_update = await db["users"].find(
        {
            "company_code": {"$ne": "None"},
            "$or": [
                {"company_name": {"$in": ["None", "", None]}},
                {"company_name": {"$exists": False}}
            ]
        }
    ).to_list(length=1000)
    
    # Also include users who have a company_code that doesn't match their company_name
    potential_mismatches = await db["users"].find(
        {
            "company_code": {"$ne": "None"},
            "company_name": {"$ne": None},
            "company_name": {"$ne": ""},
            "company_name": {"$ne": "None"}
        }
    ).to_list(length=1000)
    
    # Process all found users
    updated_count = 0
    not_found_count = 0
    no_change_count = 0
    updated_users = []
    
    # First process users with missing company name
    for user in users_to_update:
        company_code = user.get("company_code")
        
        # Find company with this code
        company = await db["companies"].find_one({"company_code": company_code})
        
        if company:
            # Update the user's company name
            await db["users"].update_one(
                {"_id": user["_id"]},
                {"$set": {"company_name": company["name"]}}
            )
            updated_count += 1
            updated_users.append({
                "email": user.get("email"),
                "username": user.get("username"),
                "old_company_name": user.get("company_name", "None"),
                "new_company_name": company["name"],
                "company_code": company_code
            })
        else:
            not_found_count += 1
    
    # Then process potential mismatches
    for user in potential_mismatches:
        company_code = user.get("company_code")
        current_company_name = user.get("company_name")
        
        # Find company with this code
        company = await db["companies"].find_one({"company_code": company_code})
        
        if company and company["name"] != current_company_name:
            # Update the user's company name
            await db["users"].update_one(
                {"_id": user["_id"]},
                {"$set": {"company_name": company["name"]}}
            )
            updated_count += 1
            updated_users.append({
                "email": user.get("email"),
                "username": user.get("username"),
                "old_company_name": current_company_name,
                "new_company_name": company["name"],
                "company_code": company_code
            })
        elif company:
            no_change_count += 1
        else:
            not_found_count += 1
    
    logger.info(f"Completed company name sync: {updated_count} users updated, {not_found_count} company codes not found, {no_change_count} already correct")
    
    return {
        "success": True,
        "updated_count": updated_count,
        "not_found_count": not_found_count,
        "no_change_count": no_change_count,
        "updated_users": updated_users
    }

# Admin endpoint to list all admin users
@router.get("/admins", status_code=status.HTTP_200_OK)
async def list_admins(current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Get a list of all admin users in the system"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can view the list of admins"
        )
    
    # Find all admin users
    admin_users = await db["users"].find(
        {"is_admin": True},
        {
            "_id": 1,
            "username": 1,
            "email": 1,
            "full_name": 1,
            "created_at": 1,
            "promoted_by": 1,
            "promoted_at": 1,
            "system_initialized": 1,
            "initialization_date": 1
        }
    ).to_list(length=100)  # Limit to 100 results
    
    # Format response
    formatted_admins = []
    for admin in admin_users:
        formatted_admins.append({
            "id": str(admin["_id"]),
            "email": admin["email"],
            "username": admin["username"],
            "full_name": admin.get("full_name"),
            "created_at": admin.get("created_at"),
            "promoted_by": admin.get("promoted_by", "System Initialization" if admin.get("system_initialized") else "Unknown"),
            "promoted_at": admin.get("promoted_at", admin.get("initialization_date", admin.get("created_at"))),
            "is_initial_admin": admin.get("system_initialized", False)
        })
    
    return {
        "count": len(formatted_admins),
        "admins": formatted_admins
    }

# Admin endpoint to revoke user verification status
@router.post("/revoke-verification", status_code=status.HTTP_200_OK)
async def revoke_user_verification(email: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to revoke a user's verified status"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can revoke user verification"
        )
    
    # Find the user by email
    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user is already unverified
    if not user.get("is_verified", False):
        return {
            "message": "User is already unverified",
            "email": email,
            "username": user.get("username"),
            "is_verified": False
        }
    
    # Prevent revoking verification of admin users
    if user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke verification of admin users"
        )
    
    # Update user verification status
    await db["users"].update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "is_verified": False,
                "verification_revoked_by": current_user["username"],
                "verification_revoked_at": datetime.utcnow()
            }
        }
    )
    
    logger.info(f"User '{email}' verification revoked by admin '{current_user['username']}'")
    
    return {
        "message": "User verification revoked successfully",
        "email": email,
        "username": user.get("username"),
        "is_verified": False
    }

# Admin endpoint to delete a user
@router.delete("/delete-user", status_code=status.HTTP_200_OK)
async def delete_user(email: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to delete a user from the system"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete users"
        )
    
    # Find the user by email
    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent deleting admin users
    if user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete admin users. Revoke admin status first."
        )
    
    # Delete user
    result = await db["users"].delete_one({"_id": user["_id"]})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
    
    logger.info(f"User '{email}' deleted by admin '{current_user['username']}'")
    
    return {
        "message": "User deleted successfully",
        "email": email,
        "username": user.get("username")
    }

# Admin endpoint to delete a company
@router.delete("/delete-company", status_code=status.HTTP_200_OK)
async def delete_company(company_code: str, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to delete a company from the system"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete companies"
        )
    
    # Find the company by code
    company = await db["companies"].find_one({"company_code": company_code})
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found"
        )
    
    # Check if there are users associated with this company
    users_count = await db["users"].count_documents({"company_code": company_code})
    if users_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete company that has {users_count} associated users. Update or delete users first."
        )
    
    # Delete company
    result = await db["companies"].delete_one({"_id": company["_id"]})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete company"
        )
    
    logger.info(f"Company '{company.get('name')}' ({company_code}) deleted by admin '{current_user['username']}'")
    
    return {
        "message": "Company deleted successfully",
        "company_code": company_code,
        "company_name": company.get("name")
    }

# Admin endpoint to create a company
@router.post("/create-company", status_code=status.HTTP_201_CREATED)
async def create_company(company: CompanyCreate, current_user: dict = Depends(get_current_user), db = Depends(get_database)):
    """Admin endpoint to create a new company with secure handling of database credentials"""
    # Check if current user is an admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create companies"
        )
    
    try:
        # Check if company name already exists (case-insensitive)
        import re
        name_pattern = re.compile(f"^{re.escape(company.name)}$", re.IGNORECASE)
        existing_company = await db["companies"].find_one({"name": {"$regex": name_pattern}})
        if existing_company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A company with this name already exists"
            )
        
        # Generate unique company code
        company_code = generate_company_code(length=8)
        
        # Create company ID
        company_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Prepare company data
        company_data = {
            "_id": company_id,
            "name": company.name,
            "company_code": company_code,
            "description": company.description,
            "address": company.address,
            "phone": company.phone,
            "industry": company.industry,
            "created_at": created_at,
            "created_by": current_user["username"],
            "is_active": True
        }
        
        # Securely handle database credentials if provided
        if company.issabel_db_host:
            company_data["issabel_db_host"] = company.issabel_db_host
            company_data["issabel_db_user"] = company.issabel_db_user
            company_data["issabel_db_name"] = company.issabel_db_name
            company_data["issabel_db_port"] = company.issabel_db_port
            
            # Encrypt password if provided
            if company.issabel_db_password:
                company_data["issabel_db_password"] = encrypt_db_password(company.issabel_db_password)
        
        # Insert into MongoDB
        await db["companies"].insert_one(company_data)
        
        logger.info(f"Company '{company.name}' created by admin '{current_user['username']}'")
        
        # Return success response
        return {
            "id": company_id,
            "name": company.name,
            "company_code": company_code,
            "description": company.description,
            "created_at": created_at
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating company: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create company: {str(e)}"
        ) 