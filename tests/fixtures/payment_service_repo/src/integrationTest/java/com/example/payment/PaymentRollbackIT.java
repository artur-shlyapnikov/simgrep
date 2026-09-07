package com.example.payment;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(SpringExtension.class)
@SpringBootTest
public class PaymentRollbackIT {

    @Autowired
    private com.example.payment.service.PaymentRollbackService rollbackService;

    @Test
    public void testRollbackPaymentSuccess() {
        Long paymentId = 12345L;
        String reason = "Payment gateway timeout";
        boolean result = rollbackService.canRollback(paymentId);
        assertTrue(result);
    }

    @Test
    public void testCannotRollbackAfterSettlement() {
    }
}