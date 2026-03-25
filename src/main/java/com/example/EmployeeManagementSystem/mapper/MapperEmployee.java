package com.example.EmployeeManagementSystem.mapper;

import com.example.EmployeeManagementSystem.dto.DtoEmployee;
import com.example.EmployeeManagementSystem.model.Employee;

public class MapperEmployee {
    public static Employee mapTOEmployee(DtoEmployee dtoEmployee){
         return new Employee(
                dtoEmployee.getId(),
                dtoEmployee.getFirstname(),
                dtoEmployee.getLastname(),
                dtoEmployee.getEmail()
        );
    }
    public static DtoEmployee mapToDtoEmployee(Employee employee){
        return new DtoEmployee(
                employee.getId(),
                employee.getFirstname(),
                employee.getLastname(),
                employee.getEmail()
        );
    }

}
