package com.example.EmployeeManagementSystem.model;

import jakarta.persistence.*;



@Entity
@Table(name = "employees")
public class Employee {
    @Id
    private int id;
    @Column(name = "first_name",nullable = false)
    private String firstname;
    @Column(name ="last_name")
    private String lastname;
    @Column(name = "email",nullable = false)
    private String email;

    public Employee() {
    }

    public Employee(int id, String firstname, String lastname, String email) {
        this.id = id;
        this.firstname = firstname;
        this.lastname = lastname;
        this.email = email;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setFirstname(String firstname) {
        this.firstname = firstname;
    }

    public void setLastname(String lastname) {
        this.lastname = lastname;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getFirstname() {
        return firstname;
    }

    public String getLastname() {
        return lastname;
    }

    public String getEmail() {
        return email;
    }
}
