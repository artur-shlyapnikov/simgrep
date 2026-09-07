# Tax Invoice Application

This is a Spring Boot application for managing tax invoices.

## Features

- Create and manage tax invoices
- Handle payment processing
- Generate reports in PDF and Excel formats
- Integration with external tax authorities

## Getting Started

1. Clone the repository
2. Run `./mvnw spring-boot:run`
3. Access OpenAPI docs at http://localhost:8080/swagger-ui.html

## Configuration

See `application.yml` for configuration options.

## Testing

Run `mvn test` to execute unit tests.

## Invoice Processing

The invoice processing system handles:
- Tax calculation
- Payment rollback on failure
- PDF generation
- Email notifications