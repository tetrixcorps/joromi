resource "aws_api_gateway_rest_api" "main" {
  name = "${var.environment}-ml-api"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "inference" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "inference"
}

# Add more API Gateway resources and methods as needed 