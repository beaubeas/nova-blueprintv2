package com.example.EmployeeManagementSystem.service;

import com.example.EmployeeManagementSystem.dto.DtoEmployee;

import java.util.List;

public interface ServiceEmployeeIn {
     DtoEmployee createEmployee(DtoEmployee dtoEmployee);

    DtoEmployee findById(int id);

    List<DtoEmployee> findAllEmployee();

    DtoEmployee updateEmployee(int id, DtoEmployee updatedemployee);

    void deleteEmployee(int id);
}
