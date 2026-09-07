package com.example.payment.service;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class PaymentRollbackService {

    @Transactional
    public void rollbackPayment(Long paymentId, String reason) {
    }

    public boolean canRollback(Long paymentId) {
        return true;
    }
}