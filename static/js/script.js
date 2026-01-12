// ✅ APPROVED DEMO (Green)
function fillApproveDemo() {
    document.querySelector("[name='Gender']").value = 1;
    document.querySelector("[name='Married']").value = 1;
    document.querySelector("[name='Dependents']").value = 0;
    document.querySelector("[name='Education']").value = 1;
    document.querySelector("[name='Self_Employed']").value = 0;
    document.querySelector("[name='ApplicantIncome']").value = 5000;
    document.querySelector("[name='CoapplicantIncome']").value = 1500;
    document.querySelector("[name='LoanAmount']").value = 150;
    document.querySelector("[name='Loan_Amount_Term']").value = 360;
    document.querySelector("[name='Credit_History']").value = 1; // ✅
    document.querySelector("[name='Property_Area']").value = 2;
}

// ❌ REJECT DEMO (Red)
function fillRejectDemo() {
    document.querySelector("[name='Gender']").value = 0;
    document.querySelector("[name='Married']").value = 0;
    document.querySelector("[name='Dependents']").value = 4;
    document.querySelector("[name='Education']").value = 0;
    document.querySelector("[name='Self_Employed']").value = 1;
    document.querySelector("[name='ApplicantIncome']").value = 500;
    document.querySelector("[name='CoapplicantIncome']").value = 0;
    document.querySelector("[name='LoanAmount']").value = 600;
    document.querySelector("[name='Loan_Amount_Term']").value = 360;
    document.querySelector("[name='Credit_History']").value = 0; // ❌ MOST IMPORTANT
    document.querySelector("[name='Property_Area']").value = 0;
}

// 🧹 CLEAR FORM
function clearForm() {
    document.querySelector("form").reset();
}
