import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SendButton } from './send-button';

describe('SendButton', () => {
  let component: SendButton;
  let fixture: ComponentFixture<SendButton>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SendButton],
    }).compileComponents();

    fixture = TestBed.createComponent(SendButton);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
