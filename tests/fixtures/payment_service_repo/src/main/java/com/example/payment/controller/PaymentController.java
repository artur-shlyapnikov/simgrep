package com.example.payment.controller;

import org.springframework.web.bind.annotation.*;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api/v1/payments")
@Tag(name = "Payment", description = "Payment management operations")
public class PaymentController {

    @GetMapping("/{id}")
    @Operation(summary = "Get payment by ID", description = "Retrieves a payment with full details")
    public Payment getPayment(@PathVariable Long id) {
        return null;
    }

    @PostMapping
    @Operation(summary = "Create new payment", description = "Creates a new payment")
    public Payment createPayment(@RequestBody PaymentRequest request) {
        return null;
    }

    @PostMapping("/{id}/rollback")
    @Operation(summary = "Rollback payment", description = "Handles payment rollback when processing fails")
    public RollbackResult rollbackPayment(@PathVariable Long id) {
        return null;
    }

    @GetMapping("/openapi.json")
    @Operation(summary = "Get OpenAPI spec", description = "Returns the OpenAPI specification for this API")
    public String getOpenApiSpec() {
        return null;
    }
}