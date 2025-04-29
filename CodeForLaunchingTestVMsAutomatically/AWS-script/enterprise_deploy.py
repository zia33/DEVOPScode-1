#!/usr/bin/env python3
"""
AWS Enterprise Auto-Deployment Tool with Full S3 Logging

This script automates AWS infrastructure deployment using Terraform.
It logs all actions—Terraform execution, SNS notifications, and rollbacks—to S3.
It also includes lifecycle policies that delete logs after 90 days.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Initialize logger globally, will be properly configured later
logger = logging.getLogger(__name__)

# ==============================================================
# Configuration Management
# ==============================================================

class Config:
    """Manages script configuration including CLI args and config file."""
    def __init__(self, args):
        self.debug = args.debug
        self.region = args.region
        self.instance_count = args.instance_count
        self.rollback = args.rollback
        self.config_file = args.config_file
        self.data = self._load_config_file()

    def _load_config_file(self):
        """Load configuration from JSON file."""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Config file {self.config_file} not found. Using default values.")
                return {}
                
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AWS Enterprise Auto-Deployment Tool")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances to deploy")
    parser.add_argument("--rollback", action="store_true", help="Enable automatic rollback on failure")
    parser.add_argument("--config-file", default="config.json", help="Path to configuration file")
    return parser.parse_args()

# ==============================================================
# Logging Configuration - Local & S3 Storage
# ==============================================================

def configure_logging(debug=False, config=None):
    """Set up logging with both local storage and S3 storage."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_filename = "enterprise_deploy.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Upload log file to S3 after every major operation
    if config and config.data.get("s3_logging_enabled", False):
        upload_log_to_s3(config, log_filename)

    return logger

# ==============================================================
# AWS Client Manager
# ==============================================================

class AWSManager:
    """Manages AWS session and service clients."""
    def __init__(self, region):
        self.region = region
        self.session = self._initialize_session()
        self.clients = self._initialize_clients()

    def _initialize_session(self):
        """Initialize AWS session with error handling."""
        try:
            return boto3.Session(region_name=self.region)
        except NoCredentialsError:
            logger.error("AWS credentials not found. Ensure authentication via AWS CLI or environment variables.")
            sys.exit(1)

    def _initialize_clients(self):
        """Initialize AWS service clients."""
        try:
            return {
                "s3": self.session.client("s3"),
                "ec2": self.session.client("ec2"),
                "iam": self.session.client("iam"),
                "sns": self.session.client("sns")
            }
        except ClientError as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            sys.exit(1)

# ==============================================================
# S3 Log Storage & Lifecycle Policy Setup
# ==============================================================

def upload_log_to_s3(config, log_filename):
    """Upload logs to an S3 bucket."""
    if not config:
        logger.warning("Config is None. Skipping S3 log upload.")
        return
        
    bucket_name = config.data.get("s3_log_bucket")
    if not bucket_name:
        logger.warning("S3 bucket for logging is not configured. Logs will remain local.")
        return
        
    try:
        s3_client = AWSManager(config.region).clients["s3"]
        s3_client.upload_file(log_filename, bucket_name, log_filename)
        logger.info(f"Uploaded log file to S3 bucket: {bucket_name}/{log_filename}")
    except ClientError as e:
        logger.error(f"Failed to upload log file to S3: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during S3 log upload: {e}")

def setup_s3_lifecycle_policy(config):
    """Set up an S3 lifecycle policy to delete logs after 90 days."""
    if not config:
        logger.warning("Config is None. Skipping S3 lifecycle policy setup.")
        return
        
    bucket_name = config.data.get("s3_log_bucket")
    if not bucket_name:
        logger.warning("S3 bucket for lifecycle policy is not set. Skipping lifecycle setup.")
        return

    lifecycle_policy = {
        "Rules": [
            {
                "ID": "AutoDeleteLogsAfter90Days",
                "Prefix": "",
                "Status": "Enabled",
                "Expiration": {"Days": 90}
            }
        ]
    }

    try:
        s3_client = AWSManager(config.region).clients["s3"]
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
        logger.info("Set up lifecycle policy for S3 logs: Expire after 90 days.")
    except ClientError as e:
        logger.error(f"Failed to configure S3 lifecycle policy: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during S3 lifecycle policy setup: {e}")

# ==============================================================
# Terraform Configuration File Generation
# ==============================================================

def generate_terraform_files(config):
    """Generate Terraform files dynamically."""
    if not config:
        logger.error("Config is None. Cannot generate Terraform files.")
        return False
        
    terraform_dir = config.data.get("terraform_dir", "./terraform")
    os.makedirs(terraform_dir, exist_ok=True)

    # Generate `variables.tf`
    variables_tf_content = f"""
    variable "aws_region" {{
      default = "{config.region}"
    }}

    variable "instance_count" {{
      default = {config.instance_count}
    }}

    variable "instance_type" {{
      default = "{config.data.get('test_instance_type', 't2.micro')}"
    }}

    variable "ami_id" {{
      default = "{config.data.get('ami_id', 'ami-0c55b159cbfafe1f0')}"
    }}
    """

    variables_tf_path = os.path.join(terraform_dir, "variables.tf")
    with open(variables_tf_path, "w") as f:
        f.write(variables_tf_content.strip())

    # Generate `main.tf`
    main_tf_content = f"""
    terraform {{
      backend "s3" {{
        bucket = "{config.data.get('terraform_state_bucket', 'enterprise-auto-state')}"
        key    = "terraform.tfstate"
        region = "{config.region}"
        encrypt = true
      }}
    }}

    provider "aws" {{
      region = var.aws_region
    }}

    resource "aws_instance" "test_vm" {{
      ami           = var.ami_id
      instance_type = var.instance_type
      count         = var.instance_count

      tags = {{
        Name        = "TestVM"
        Environment = "Test"
        Project     = "EnterpriseAutoDeploy"
      }}
    }}
    """

    main_tf_path = os.path.join(terraform_dir, "main.tf")
    with open(main_tf_path, "w") as f:
        f.write(main_tf_content.strip())

    # Generate `outputs.tf`
    outputs_tf_content = """
    output "instance_ids" {
      value = aws_instance.test_vm[*].id
    }

    output "public_ips" {
      value = aws_instance.test_vm[*].public_ip
    }
    """

    outputs_tf_path = os.path.join(terraform_dir, "outputs.tf")
    with open(outputs_tf_path, "w") as f:
        f.write(outputs_tf_content.strip())

    logger.info(f"Generated Terraform configuration files at: {terraform_dir}")
    return True

# ==============================================================
# Terraform Execution
# =============================================================

def run_terraform(config):
    """Run Terraform commands to deploy instances."""
    if not generate_terraform_files(config):
        logger.error("Failed to generate Terraform files. Aborting deployment.")
        return False
        
    terraform_dir = config.data.get("terraform_dir", "./terraform")
    try:
        logger.info("Initializing Terraform...")
        subprocess.run(["terraform", "init"], cwd=terraform_dir, check=True)
        subprocess.run(["terraform", "apply", "-auto-approve"], cwd=terraform_dir, check=True)
        logger.info("Terraform deployment completed successfully.")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Terraform command failed: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Terraform execution: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return False

# ==============================================================
# SNS Notification Handling
# ==============================================================

def send_sns_email(config, subject, message):
    """Send an email notification via AWS SNS and log the event to S3."""
    if not config:
        logger.warning("Config is None. Skipping SNS email notification.")
        return
        
    sns_topic_arn = config.data.get("sns_topic_arn")
    if not sns_topic_arn:
        logger.warning("SNS topic ARN not set. Email notifications will be skipped.")
        return
    
    try:
        sns_client = AWSManager(config.region).clients["sns"]
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=message,
            Subject=subject
        )
        logger.info(f"Email notification sent: {subject}")
        upload_log_to_s3(config, "enterprise_deploy.log")
    except ClientError as e:
        logger.error(f"Failed to send email notification: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")
    except Exception as e:
        logger.error(f"Unexpected error during SNS email notification: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")

# ==============================================================
# Rollback Mechanism
# ==============================================================

def run_terraform_destroy(config):
    """Run Terraform destroy for rollback and log the rollback event to S3."""
    if not config:
        logger.error("Config is None. Cannot perform rollback.")
        return False
        
    terraform_dir = config.data.get("terraform_dir", "./terraform")
    try:
        logger.info("Running Terraform destroy for rollback...")
        subprocess.run(["terraform", "destroy", "-auto-approve"], cwd=terraform_dir, check=True)
        logger.info("Terraform destroy completed successfully.")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Terraform destroy failed during rollback: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Terraform destroy: {e}")
        upload_log_to_s3(config, "enterprise_deploy.log")
        return False

# ==============================================================
# Main Execution
# ==============================================================

def main():
    """Main script execution."""
    args = parse_arguments()
    global logger
    # Initialize logger with basic configuration first
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create config object
    config = Config(args)
    
    # Properly configure logging with config object
    logger = configure_logging(args.debug, config)

    # Initialize AWS services & log storage policy setup
    try:
        aws_manager = AWSManager(config.region)
        setup_s3_lifecycle_policy(config)
    except Exception as e:
        logger.error(f"Failed to initialize AWS services: {e}")
        sys.exit(1)

    try:
        success = run_terraform(config)
        if success:
            logger.info("Deployment completed successfully.")
            send_sns_email(config, "Deployment Success", "Terraform deployment completed successfully.")
        else:
            logger.error("Deployment failed during Terraform execution.")
            send_sns_email(config, "Deployment Failure", "Terraform deployment failed during execution.")
            if config.rollback:
                run_terraform_destroy(config)
            sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        send_sns_email(config, "Deployment Failure", f"Terraform deployment failed: {e}")
        if config.rollback:
            run_terraform_destroy(config)
        sys.exit(1)

if __name__ == "__main__":
    main()